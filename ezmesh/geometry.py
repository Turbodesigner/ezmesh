from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Iterable, Set, Tuple, Union
import numpy as np
import numpy.typing as npt
import gmsh
from .importers import import_from_gmsh


ArrayType = Union[npt.NDArray[np.float64], Tuple[float, float]]

class DimType(Enum):
    POINT = 0
    CURVE = 1
    SURFACE = 2
    VOLUME = 3


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
    coord: npt.NDArray[np.float64]
    "coordinate of point"
    mesh_size: float
    "mesh size for point"

    def __post_init__(self):
        super().__init__()
        self.dim_type = DimType.POINT
        self.x = self.coord[0]
        self.y = self.coord[1]
        self.z = self.coord[2] if len(self.coord) == 3 else 0

    def before_sync(self):
        if not self.before_sync_initiated:
            self.tag = gmsh.model.geo.add_point(self.x, self.y, self.z, self.mesh_size)
        super().before_sync()


@dataclass
class Line(MeshTransaction):
    start: Point
    "starting point of line"
    end: Point
    "ending point of line"

    def __post_init__(self):
        self.dim_type = DimType.CURVE
        super().__init__()

    def before_sync(self):
        if not self.before_sync_initiated:
            self.start.before_sync()
            self.end.before_sync()
            self.tag = gmsh.model.geo.add_line(self.start.tag, self.end.tag)
        super().before_sync()


@dataclass
class CurveLoop(MeshTransaction):
    groups: Dict[str, Iterable[ArrayType]]
    "2D array of coordinate points"

    mesh_size: Union[float, List[float]]
    "Mesh size for points, If list, must be same length as coords."

    fields: List[Field] = field(default_factory=list)
    "fields to be added to the curve loop"

    transfinite_cell_counts: Optional[Dict[float, Union[List[Union[Line, int]], Literal["all"]]]] = None
    "mesh size for lines in surface"

    def __post_init__(self):
        super().__init__()
        self.dim_type = DimType.CURVE
        self.line_tags: List[int] = []

        if isinstance(self.mesh_size, List):
            assert len(self.mesh_size) == len(self.points)

        self.points: List[Point] = []
        self.lines: List[Line] = []
        self.markers: Dict[str, List[Line]] = {}
        added_points: Dict[Tuple, Point] = {}

        for i, (label, coords) in enumerate(self.groups.items()):
            lines: List[Line] = []
            for j, coord in enumerate(coords):
                # get mesh size for point
                mesh_size = self.mesh_size[i] if isinstance(self.mesh_size, List) else self.mesh_size

                # convert coord to numpy array if it isn't and store point lookup key
                coord = np.asarray(coord)
                point_key = tuple(coord)
                
                # create point and add to lookup if it hasn't been added, otherwise lookup point
                if point_key not in added_points:
                    point = Point(coord, mesh_size)
                    added_points[point_key] = point
                else:
                    point = added_points[point_key]

                # adding lines to connect points
                if len(self.points) > 0:
                    lines.append(Line(self.points[-1], point))
                
                # store point
                self.points.append(point)

            self.lines += lines
            self.markers[label] = lines

    def before_sync(self):
        if not self.before_sync_initiated:
            for line in self.lines:
                line.before_sync()
                assert line.tag is not None
                self.line_tags.append(line.tag)
            self.tag = gmsh.model.geo.add_curve_loop(self.line_tags)
            for field in self.fields:
                field.before_sync(self)

        super().before_sync()

    def after_sync(self):
        if not self.after_sync_initiated and self.markers:
            for (name, label_lines) in self.markers.items():
                if label_lines == "all":
                    label_line_tags = self.line_tags
                else:
                    label_line_tags = [
                        label_line.tag if isinstance(label_line, Line)
                        else self.lines[label_line].tag
                        for label_line in label_lines
                    ]
                physical_group_tag = gmsh.model.add_physical_group(DimType.CURVE.value, label_line_tags)
                gmsh.model.set_physical_name(DimType.CURVE.value, physical_group_tag, name)
        
        for field in self.fields:
            field.after_sync(self)

        if self.transfinite_cell_counts is not None:
            for (cell_count, transfinite_lines) in self.transfinite_cell_counts.items():
                for line in (self.lines if transfinite_lines == "all" else transfinite_lines):
                    gmsh.model.mesh.set_transfinite_curve(
                        line.tag if isinstance(line, Line)
                        else self.lines[line].tag,
                        cell_count+1
                    )

        super().after_sync()


@dataclass
class PlaneSurface(MeshTransaction):
    outline: CurveLoop
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
        self.curve_loops = [self.outline] + self.holes

    def before_sync(self):
        if not self.before_sync_initiated:
            curve_loop_tags = []
            for curve_loop in self.curve_loops:
                curve_loop.before_sync()
                curve_loop_tags.append(curve_loop.tag)
            self.tag = gmsh.model.geo.add_plane_surface(curve_loop_tags)

            if self.is_quad_mesh: 
                gmsh.model.mesh.set_recombine(self.tag, 2)  # type: ignore
        super().before_sync()

    def after_sync(self):
        if not self.after_sync_initiated:
            for curve_loop in self.curve_loops:
                curve_loop.after_sync()
            if self.label is not None:
                physical_group_tag = gmsh.model.add_physical_group(DimType.SURFACE.value, [self.tag])
                gmsh.model.set_physical_name(DimType.SURFACE.value, physical_group_tag, self.label)

        if self.transfinite_corners is not None:
            corner_tags = [
                corner.tag if isinstance(corner, Point)
                else self.outline.points[corner].tag
                for corner in self.transfinite_corners
            ]
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
    
    quads: bool = False
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
            if self.quads:
                gmsh.model.mesh.field.setNumber(self.tag, "Quads", int(self.quads))
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
        mesh = import_from_gmsh(gmsh.model)
        return mesh

    def write(self, filename: str):
        gmsh.write(filename)
