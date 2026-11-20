# FaSTTUBe path-planning library

- Upstream: <https://github.com/papalotis/ft-fsd-path-planning>
- Upstream version at the imported revision: `0.4.3.2`
- Imported commit: `1eb5ee2251c7e28a83fbfe2f9222d28f18e97887`
- License: MIT; see [LICENSE](LICENSE).
- The upstream package source is under [`src/fsd_path_planning`](../../src/fsd_path_planning). Two small robustness patches handle singular circle fits on collinear/straight paths: path extension follows the last non-zero tangent, and curvature calculation assigns zero curvature to straight windows. See `extend_path` in `calculate_path/core_calculate_path.py` and `calculate_path_curvature` in `calculate_path/path_parameterization.py`.

The upstream README, packaging metadata, requirements, and demo assets are retained in this directory as source context. Khonshu's adapter and safety checks are implemented separately under `src/khonshu`.
