import numpy as np

from asset_health_index.baseline import HealthyBaseline
from asset_health_index.mahalanobis import mahalanobis_distance_batch


def test_mahalanobis_batch_returns_expected_shape():
    baseline = HealthyBaseline(
        mean_vector=np.array([0.0, 0.0]),
        covariance_matrix=np.eye(2),
        inverse_covariance_matrix=np.eye(2),
    )
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    md = mahalanobis_distance_batch(x, baseline)
    assert md.shape == (2,)
    assert np.isclose(md[0], 0.0)
    assert np.isclose(md[1], 5.0)
