from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Iterable, Tuple, Union, cast
import numpy as np
import numpy.typing as npt
import gmsh
from .importers import import_from_gmsh

Number = Union[int, float]
PointCoordType = Union[npt.NDArray[np.float64], Tuple[Number, Number], List[Number]]


class DimType(Enum):
    POINT = 0
    CURVE = 1
    SURFACE = 2
    VOLUME = 3


def get_points_and_lines(
    coords: npt.NDArray[np.float64],
    mesh_size: Union[float, List[float]],
    labels: Optional[Union[List[str], str]] = None,
    cell_counts: Optional[Union[List[int], int]] = None,
    mesh_types: Optional[Union[List[str], str]] = None,
    mesh_coeffs: Optional[Union[float, List[float]]] = None,
):
    lines = []
    points: List[Point] = []
    line_index = 0
    for i in range(len(coords) + 1):
        # getting mesh size for point
        mesh_size = mesh_size[i] if isinstance(mesh_size, List) else mesh_size

        # adding points
        if i == len(coords):
            point = points[0]
        else:
            coord = coords[i]
            point = Point(coord, mesh_size)
            points.append(point)

        # adding lines to connect points
        if i > 0:
            label = (labels if isinstance(labels, str) else labels[line_index]) if labels else None
            if cell_counts:
                cell_count = cell_counts[line_index] if isinstance(cell_counts, List) else cell_counts
                mesh_type = (mesh_types[line_index] if isinstance(mesh_types, List) else mesh_types) if mesh_types else "Progression"
                mesh_coeff = (mesh_coeffs[line_index] if isinstance(mesh_coeffs, List) else mesh_coeffs) if mesh_coeffs else 1.0
                line = TransfiniteLine(start=points[i-1], end=point, label=label, cell_count=cell_count, mesh_type=mesh_type, coeff=mesh_coeff)
            else:
                line = Line(start=points[i-1], end=point, label=label)
            lines.append(line)
            line_index += 1

    return points, lines


class MeshTransaction:
    def __init__(self) -> None:
        self.tag: Optional[int] = None
        self.before_sync_initiated: bool = False
        self.after_sync_initiated: bool = False

    def before_sync(self):
        "completes transaction before syncronization and returns tag."
        self.before_sync_initiated = True

    def after_sync(self):
        "completes transaction after syncronization and returns tag."
        self.after_sync_initiated = True


class Field(MeshTransaction):
    def __init__(self) -> None:
        super().__init__()

    def before_sync(self, curve_loop: "CurveLoop"):
        super().before_sync()

    def after_sync(self, curve_loop: "CurveLoop"):
        super().after_sync()


@dataclass
class Point(MeshTransaction):
    coord: PointCoordType
    "coordinate of point"

    mesh_size: float
    "mesh size for point"

    label: Optional[int] = None
    "tag of point"

    def __post_init__(self):
        super().__init__()
        self.coord = np.asarray(self.coord)
        self.dim_type = DimType.POINT
        self.x = self.coord[0]
        self.y = self.coord[1]
        self.z = self.coord[2] if len(self.coord) == 3 else 0

    def before_sync(self):
        if not self.before_sync_initiated:
            self.tag = gmsh.model.geo.add_point(self.x, self.y, self.z, self.mesh_size)
        super().before_sync()


@dataclass(kw_only=True)
class Line(MeshTransaction):
    start: Point
    "starting point of line"

    end: Point
    "ending point of line"

    label: Optional[str] = None
    "physical group label"

    def __post_init__(self):
        super().__init__()
        self.dim_type = DimType.CURVE

    def before_sync(self):
        if not self.before_sync_initiated:
            self.start.before_sync()
            self.end.before_sync()
            self.tag = gmsh.model.geo.add_line(self.start.tag, self.end.tag)
        super().before_sync()


@dataclass(kw_only=True)
class TransfiniteLine(Line):
    cell_count: int
    "number of cells in the line"

    mesh_type: str = "Progression"
    "mesh type for the line"

    coeff: float = 1.0
    "coefficient for the line"

    def __post_init__(self):
        super().__post_init__()

    def after_sync(self):
        if not self.after_sync_initiated:
            gmsh.model.mesh.set_transfinite_curve(
                self.tag,
                self.cell_count+1,
                self.mesh_type,
                self.coeff
            )
        super().after_sync()


@dataclass(kw_only=True)
class CurveLoop(MeshTransaction):
    lines: List[Line]
    "Lines of curve"

    fields: List[Field] = field(default_factory=list)
    "fields to be added to the curve loop"

    def __post_init__(self):
        super().__init__()
        self.dim_type = DimType.CURVE
        self.line_tags: List[int] = []
        self.points = [line.start for line in self.lines]

    def before_sync(self):
        if not self.before_sync_initiated:
            for line in self.lines:
                line.before_sync()
                self.line_tags.append(cast(int, line.tag))
            self.tag = gmsh.model.geo.add_curve_loop(self.line_tags)
            for field in self.fields:
                field.before_sync(self)

        super().before_sync()

    def after_sync(self):
        if not self.after_sync_initiated:

            line_group_tags: Dict[str, List[int]] = {}
            for line in self.lines:
                if line.label:
                    if line.label not in line_group_tags:
                        line_group_tags[line.label] = []
                    line_group_tags[line.label].append(cast(int, line.tag))
                line.after_sync()

            for (label, group_tags) in line_group_tags.items():
                physical_group_tag = gmsh.model.add_physical_group(DimType.CURVE.value, group_tags)
                gmsh.model.set_physical_name(DimType.CURVE.value, physical_group_tag, label)

            for field in self.fields:
                field.after_sync(self)

        super().after_sync()


@dataclass(kw_only=True)
class PlaneSurface(MeshTransaction):
    outlines: List[CurveLoop]
    "outline curve loop that make up the surface"

    holes: List[CurveLoop] = field(default_factory=list)
    "hole curve loops that make up the surface"

    label: Optional[str] = None
    "label for physical group surface"

    is_quad_mesh: bool = False
    "if true, surface mesh is made of quadralateral cells, else triangular cells"

    transfinite_corners: Optional[List[Union[Point, int]]] = None
    "corners of transfinite surface"

    def __post_init__(self):
        super().__init__()
        self.dim_type = DimType.SURFACE
        self.curve_loops = self.outlines + self.holes

    def before_sync(self):
        if not self.before_sync_initiated:
            curve_loop_tags = []
            for curve_loop in self.curve_loops:
                curve_loop.before_sync()
                curve_loop_tags.append(curve_loop.tag)
            self.tag = gmsh.model.geo.add_plane_surface(curve_loop_tags)

        super().before_sync()

    def after_sync(self):
        if not self.after_sync_initiated:
            for curve_loop in self.curve_loops:
                curve_loop.after_sync()
            if self.label is not None:
                physical_group_tag = gmsh.model.add_physical_group(DimType.SURFACE.value, [self.tag])
                gmsh.model.set_physical_name(DimType.SURFACE.value, physical_group_tag, self.label)

            if self.is_quad_mesh:
                gmsh.model.mesh.set_recombine(2, self.tag)  # type: ignore
        super().after_sync()


@dataclass(kw_only=True)
class TransfinitePlaneSurface(PlaneSurface):
    corners: List[Point]

    def __post_init__(self):
        return super().__post_init__()

    def after_sync(self):
        if not self.after_sync_initiated:
            corner_tags: List[int] = []
            for corner in self.corners:
                corner_tags.append(cast(int, corner.tag))
            gmsh.model.mesh.set_transfinite_surface(self.tag, cornerTags=corner_tags)

        super().after_sync()


@dataclass
class BoundaryLayer(Field):
    aniso_max: Optional[float] = None
    "threshold angle for creating a mesh fan in the boundary layer"

    hfar: Optional[float] = None
    "element size far from the wall"

    hwall_n: Optional[float] = None
    "mesh Size Normal to the The Wal"

    ratio: Optional[float] = None
    "size Ratio Between Two Successive Layers"

    thickness: Optional[float] = None
    "maximal thickness of the boundary layer"

    intersect_metrics: bool = False
    "intersect metrics of all surfaces"

    is_quad_mesh: bool = False
    "generate recombined elements in the boundary layer"

    def __post_init__(self):
        super().__init__()

    def after_sync(self, curve_loop: CurveLoop):
        if not self.after_sync_initiated:
            self.tag = gmsh.model.mesh.field.add('BoundaryLayer')
            gmsh.model.mesh.field.setNumbers(self.tag, 'CurvesList', curve_loop.line_tags)
            if self.aniso_max:
                gmsh.model.mesh.field.setNumber(self.tag, "AnisoMax", self.aniso_max)
            if self.intersect_metrics:
                gmsh.model.mesh.field.setNumber(self.tag, "IntersectMetrics", self.intersect_metrics)
            if self.is_quad_mesh:
                gmsh.model.mesh.field.setNumber(self.tag, "Quads", int(self.is_quad_mesh))
            if self.hfar:
                gmsh.model.mesh.field.setNumber(self.tag, "hfar", self.hfar)
            if self.hwall_n:
                gmsh.model.mesh.field.setNumber(self.tag, "hwall_n", self.hwall_n)
            if self.ratio:
                gmsh.model.mesh.field.setNumber(self.tag, "ratio", self.ratio)
            if self.thickness:
                gmsh.model.mesh.field.setNumber(self.tag, "thickness", self.thickness)

            gmsh.model.mesh.field.setAsBoundaryLayer(self.tag)
        super().after_sync(curve_loop)


class Geometry:
    def __enter__(self):
        gmsh.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        gmsh.finalize()

    def generate(self, transactions: Union[MeshTransaction, List[MeshTransaction]]):
        if isinstance(transactions, list):
            for transaction in transactions:
                transaction.before_sync()
        else:
            transactions.before_sync()
        gmsh.model.geo.synchronize()
        if isinstance(transactions, list):
            for transaction in transactions:
                transaction.after_sync()
        else:
            transactions.after_sync()
        gmsh.model.mesh.generate()
        gmsh.option.set_number("Mesh.SaveAll", 1)
        return import_from_gmsh(gmsh.model)

    def write(self, filename: str):
        gmsh.write(filename)
