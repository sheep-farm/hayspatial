# hayspatial

Spatial econometrics plugin for Hayashi.

## Installation

```bash
hay install sheep-farm/hayspatial
```

## Usage

```hayashi
import("sheep-farm/hayspatial", as=spatial)

// Calculate Moran's I spatial autocorrelation
let values = [1.0, 2.0, 3.0, 4.0, 5.0]
let weights = [[0.0, 0.5, 0.5, 0.0, 0.0], [0.5, 0.0, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.5, 0.0]]
let moran = spatial::morans_i(values, weights)
print(moran)

// Calculate distance matrix
let x = [0.0, 1.0, 2.0]
let y = [0.0, 0.0, 0.0]
let dist = spatial::distance_matrix(x, y)
print(dist)

// Create inverse distance weights
let idw = spatial::inverse_distance_weights(x, y, 1.0)
print(idw)
```

## Functions

### Spatial Autocorrelation
- `morans_i(values, weights)` - Moran's I statistic
- `gearys_c(values, weights)` - Geary's C statistic
- `spatial_autocorrelation(values, weights, lag)` - Spatial autocorrelation at lag
- `spatial_correlogram(values, weights, max_lag)` - Spatial correlogram

### Spatial Weights
- `distance_matrix(x_coords, y_coords)` - Euclidean distance matrix
- `inverse_distance_weights(x_coords, y_coords, alpha)` - Inverse distance weights
- `spatial_weights_knn(x_coords, y_coords, k)` - k-nearest neighbors weights

### Spatial Analysis
- `spatial_lag(values, weights)` - Spatial lag of values
- `local_moran(values, weights)` - Local Moran's I
- `getis_ord_g(values, weights)` - Getis-Ord Gi* statistic

## Development

```bash
cargo build --release
cp target/release/libhayspatial.so ~/.hay/packages/sheep-farm/hayspatial.so
```

## Dependencies

- Uses only hayashi-plugin-sdk (no external dependencies)
