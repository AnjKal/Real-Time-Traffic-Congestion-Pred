from dataclasses import dataclass

from src.config import BBOX


@dataclass(frozen=True)
class Node:
    node_id: int
    name: str
    lat: float
    lon: float


NODES = [
    Node(0,  "Hebbal Flyover",         13.0355, 77.5970),
    Node(1,  "Mekhri Circle",          13.0210, 77.5820),
    Node(2,  "Yeshwanthpur Circle",    13.0280, 77.5410),
    Node(3,  "Majestic",               12.9774, 77.5713),
    Node(4,  "M.G. Road Trinity",      12.9756, 77.6050),
    Node(5,  "Richmond Circle",        12.9627, 77.5950),
    Node(6,  "Shivajinagar",           12.9850, 77.6050),
    Node(7,  "Indiranagar 100Ft",      12.9719, 77.6412),
    Node(8,  "Domlur Flyover",         12.9611, 77.6387),
    Node(9,  "Koramangala Sony",       12.9352, 77.6245),
    Node(10, "Silk Board", 12.9165, 77.6220),    Node(11, "BTM Layout",             12.9166, 77.6101),
    Node(12, "Jayanagar 4th Block",    12.9250, 77.5938),
    Node(13, "Marathahalli Bridge",    12.9583, 77.6975),
    Node(14, "Bellandur",              12.9340, 77.6780),
    Node(15, "ORR Sarjapur",           12.9280, 77.6880),
    Node(16, "HSR Agara",              12.9150, 77.6380),
    Node(17, "KR Puram",               12.9981, 77.6950),
    Node(18, "Tin Factory",            12.9975, 77.6760),
    Node(19, "Whitefield ITPL",        12.9698, 77.7500),
]
assert len(NODES) == 20
assert len({n.node_id for n in NODES}) == 20
for n in NODES:
    assert BBOX["south"] <= n.lat <= BBOX["north"]
    assert BBOX["west"] <= n.lon <= BBOX["east"]
