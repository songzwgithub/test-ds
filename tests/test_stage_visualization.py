
from pypsds.monitoring.stage_visualization import STAGES, NUM, STAGE_NAMES, STAGE_NUMBER

def test_visualization_registry():
    assert len(STAGES) == 39
    assert STAGES == STAGE_NAMES
    assert NUM == STAGE_NUMBER
    assert NUM["phase_linking"] == 4
    assert NUM["point_products"] == 39
