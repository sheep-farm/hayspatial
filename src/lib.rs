#![allow(clippy::not_unsafe_ptr_arg_deref)]
use hayashi_plugin_sdk::value::Geometry;
use hayashi_plugin_sdk::{hayashi_fn, hayashi_plugin};

hayashi_plugin!();

/// 1. morans_i(values, weights)
/// Moran's I spatial autocorrelation statistic
/// values: variable values
/// weights: spatial weights matrix (simplified as list)
#[hayashi_fn]
pub fn morans_i(values: Vec<f64>, weights: Vec<Vec<f64>>) -> f64 {
    let n = values.len();
    if n == 0 || weights.is_empty() {
        return 0.0;
    }

    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let numerator: f64 = values
        .iter()
        .enumerate()
        .map(|(i, &yi)| {
            weights
                .iter()
                .take(n)
                .map(|row| {
                    if let Some(&wj) = row.get(i) {
                        let yj = values.get(i).unwrap_or(&0.0);
                        wj * (yj - mean) * (yi - mean)
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
        })
        .sum();

    let denominator: f64 = values.iter().map(|&y| (y - mean).powi(2)).sum();

    if denominator == 0.0 {
        return 0.0;
    }

    let total_weights: f64 = weights.iter().map(|row| row.iter().sum::<f64>()).sum();
    numerator / denominator * (n as f64 / total_weights)
}

/// 2. gearys_c(values, weights)
/// Geary's C spatial autocorrelation statistic
/// values: variable values
/// weights: spatial weights matrix
#[hayashi_fn]
pub fn gearys_c(values: Vec<f64>, weights: Vec<Vec<f64>>) -> f64 {
    let n = values.len();
    if n == 0 || weights.is_empty() {
        return 1.0;
    }

    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let numerator: f64 = values
        .iter()
        .enumerate()
        .map(|(i, &yi)| {
            weights
                .iter()
                .take(n)
                .map(|row| {
                    if let Some(&wj) = row.get(i) {
                        let yj = values.get(i).unwrap_or(&0.0);
                        wj * (yj - yi).powi(2)
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
        })
        .sum();

    let denominator: f64 = values.iter().map(|&y| (y - mean).powi(2)).sum();

    if denominator == 0.0 {
        return 1.0;
    }

    let total_weights: f64 = weights.iter().map(|row| row.iter().sum::<f64>()).sum();
    (n as f64 / total_weights) * numerator / denominator
}

/// 3. spatial_lag(values, weights)
/// Spatial lag of values
/// values: variable values
/// weights: spatial weights matrix
#[hayashi_fn]
pub fn spatial_lag(values: Vec<f64>, weights: Vec<Vec<f64>>) -> Vec<f64> {
    let n = values.len();
    if n == 0 || weights.is_empty() {
        return vec![0.0; n];
    }

    values
        .iter()
        .enumerate()
        .map(|(i, _yi)| {
            weights
                .iter()
                .take(n)
                .map(|row| {
                    if let Some(&wj) = row.get(i) {
                        let yj = values.get(i).unwrap_or(&0.0);
                        wj * yj
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
        })
        .collect()
}

/// 4. distance_matrix(x_coords, y_coords)
/// Calculate Euclidean distance matrix
/// x_coords: x coordinates
/// y_coords: y coordinates
#[hayashi_fn]
pub fn distance_matrix(x_coords: Vec<f64>, y_coords: Vec<f64>) -> Vec<Vec<f64>> {
    let n = x_coords.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            let dx = x_coords[i] - x_coords[j];
            let dy = y_coords[i] - y_coords[j];
            matrix[i][j] = (dx * dx + dy * dy).sqrt();
        }
    }

    matrix
}

/// 5. inverse_distance_weights(x_coords, y_coords, alpha)
/// Create inverse distance weights matrix
/// x_coords: x coordinates
/// y_coords: y coordinates
/// alpha: distance decay parameter
#[hayashi_fn]
pub fn inverse_distance_weights(
    x_coords: Vec<f64>,
    y_coords: Vec<f64>,
    alpha: f64,
) -> Vec<Vec<f64>> {
    let n = x_coords.len();
    let mut weights = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let dx = x_coords[i] - x_coords[j];
                let dy = y_coords[i] - y_coords[j];
                let distance = (dx * dx + dy * dy).sqrt();
                if distance > 0.0 {
                    weights[i][j] = 1.0 / distance.powf(alpha);
                }
            }
        }
    }

    // Row normalize
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let row_sum: f64 = weights[i].iter().sum();
        if row_sum > 0.0 {
            for j in 0..n {
                weights[i][j] /= row_sum;
            }
        }
    }

    weights
}

/// 6. local_moran(values, weights)
/// Local Moran's I for each observation
/// values: variable values
/// weights: spatial weights matrix
#[hayashi_fn]
pub fn local_moran(values: Vec<f64>, weights: Vec<Vec<f64>>) -> Vec<f64> {
    let n = values.len();
    if n == 0 || weights.is_empty() {
        return vec![0.0; n];
    }

    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let variance: f64 = values.iter().map(|&y| (y - mean).powi(2)).sum::<f64>() / n as f64;

    if variance == 0.0 {
        return vec![0.0; n];
    }

    values
        .iter()
        .enumerate()
        .map(|(i, _yi)| {
            let local_sum: f64 = weights
                .iter()
                .take(n)
                .map(|row| {
                    if let Some(&wj) = row.get(i) {
                        let yj = values.get(i).unwrap_or(&0.0);
                        wj * (yj - mean)
                    } else {
                        0.0
                    }
                })
                .sum::<f64>();

            let yi = values.get(i).unwrap_or(&0.0);
            (yi - mean) / variance * local_sum
        })
        .collect()
}

/// 7. getis_ord_g(values, weights)
/// Getis-Ord Gi* statistic for local spatial association
/// values: variable values
/// weights: spatial weights matrix
#[hayashi_fn]
pub fn getis_ord_g(values: Vec<f64>, weights: Vec<Vec<f64>>) -> Vec<f64> {
    let n = values.len();
    if n == 0 || weights.is_empty() {
        return vec![0.0; n];
    }

    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let s2: f64 = values.iter().map(|&y| (y - mean).powi(2)).sum::<f64>() / n as f64;

    if s2 == 0.0 {
        return vec![0.0; n];
    }

    values
        .iter()
        .enumerate()
        .map(|(i, _yi)| {
            let local_sum: f64 = weights
                .iter()
                .take(n)
                .map(|row| {
                    if let Some(&wj) = row.get(i) {
                        let yj = values.get(i).unwrap_or(&0.0);
                        wj * yj
                    } else {
                        0.0
                    }
                })
                .sum::<f64>();

            let local_mean = local_sum
                / weights
                    .iter()
                    .take(n)
                    .map(|row| row.iter().sum::<f64>())
                    .sum::<f64>();

            (local_mean - mean)
                / (s2.sqrt()
                    * ((weights
                        .iter()
                        .take(n)
                        .map(|row| row.iter().sum::<f64>())
                        .sum::<f64>()
                        - 1.0)
                        / (n - 1) as f64)
                        .sqrt())
        })
        .collect()
}

/// 8. spatial_autocorrelation(values, weights, lag)
/// Spatial autocorrelation at different lags
/// values: time series values
/// weights: spatial weights
/// lag: lag order
#[hayashi_fn]
pub fn spatial_autocorrelation(values: Vec<f64>, weights: Vec<Vec<f64>>, lag: i64) -> f64 {
    let n = values.len();
    let lag = lag as usize;
    if n <= lag || weights.is_empty() {
        return 0.0;
    }

    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let numerator: f64 = values
        .iter()
        .skip(lag)
        .enumerate()
        .map(|(i, &yi)| {
            let yj = values[i];
            (yi - mean) * (yj - mean)
        })
        .sum();

    let denominator: f64 = values.iter().map(|&y| (y - mean).powi(2)).sum();

    if denominator == 0.0 {
        return 0.0;
    }

    numerator / denominator
}

/// 9. spatial_weights_knn(x_coords, y_coords, k)
/// Create k-nearest neighbors spatial weights
/// x_coords: x coordinates
/// y_coords: y coordinates
/// k: number of neighbors
#[hayashi_fn]
pub fn spatial_weights_knn(x_coords: Vec<f64>, y_coords: Vec<f64>, k: i64) -> Vec<Vec<f64>> {
    let n = x_coords.len();
    let k = k.min(n as i64 - 1) as usize;
    let mut weights = vec![vec![0.0; n]; n];

    for i in 0..n {
        let mut distances: Vec<(usize, f64)> = (0..n)
            .map(|j| {
                if i == j {
                    (j, f64::INFINITY)
                } else {
                    let dx = x_coords[i] - x_coords[j];
                    let dy = y_coords[i] - y_coords[j];
                    (j, (dx * dx + dy * dy).sqrt())
                }
            })
            .collect();

        distances.sort_by(|a, b| a.1.total_cmp(&b.1));

        for (idx, _) in distances.iter().take(k) {
            weights[i][*idx] = 1.0 / k as f64;
        }
    }

    weights
}

/// 10. spatial_correlogram(values, weights, max_lag)
/// Spatial correlogram
/// values: variable values
/// weights: spatial weights
/// max_lag: maximum lag
#[hayashi_fn]
pub fn spatial_correlogram(values: Vec<f64>, weights: Vec<Vec<f64>>, max_lag: i64) -> Vec<f64> {
    (0..=max_lag)
        .map(|lag| {
            let n = values.len();
            let lag = lag as usize;
            if n <= lag || weights.is_empty() {
                return 0.0;
            }

            let mean: f64 = values.iter().sum::<f64>() / n as f64;
            let numerator: f64 = values
                .iter()
                .skip(lag)
                .enumerate()
                .map(|(i, &yi)| {
                    let yj = values[i];
                    (yi - mean) * (yj - mean)
                })
                .sum();

            let denominator: f64 = values.iter().map(|&y| (y - mean).powi(2)).sum();

            if denominator == 0.0 {
                return 0.0;
            }

            numerator / denominator
        })
        .collect()
}

/// wkt_to_coords(geometries) -> {"x": Vec<f64>, "y": Vec<f64>}
///
/// Extrai as coordenadas lon/lat de uma lista de geometrias WKT `POINT(lon lat)`.
/// Retorna dois vetores paralelos `x` (longitude) e `y` (latitude), prontos para
/// uso em `distance_matrix`, `spatial_weights_knn` e demais funções hayspatial.
///
/// Geometrias que não são POINT (polígonos, linhas) retornam o centroide da
/// bounding box como ponto representativo.
///
/// # Exemplo (.hay)
/// ```hay
/// let estados = haygeobr::read_state({})
/// let geoms = haygeobr::geometry_col(estados)
/// let coords = hayspatial::wkt_to_coords(geoms)
/// let W = hayspatial::spatial_weights_knn(coords["x"], coords["y"], 4)
/// ```
#[hayashi_fn]
pub fn wkt_to_coords(geometries: Vec<Geometry>) -> std::collections::HashMap<String, Vec<f64>> {
    let mut xs = Vec::with_capacity(geometries.len());
    let mut ys = Vec::with_capacity(geometries.len());

    for geom in &geometries {
        let wkt = geom.wkt().trim();
        let (x, y) = parse_wkt_centroid(wkt);
        xs.push(x);
        ys.push(y);
    }

    let mut out = std::collections::HashMap::new();
    out.insert("x".to_string(), xs);
    out.insert("y".to_string(), ys);
    out
}

/// Extrai um ponto representativo (centroide aproximado) de WKT.
fn parse_wkt_centroid(wkt: &str) -> (f64, f64) {
    // POINT(lon lat) ou POINT (lon lat)
    if let Some(rest) = wkt.to_uppercase().strip_prefix("POINT") {
        let inner = rest.trim().trim_matches(['(', ')']).trim();
        let mut parts = inner.split_whitespace();
        let x = parts.next().and_then(|s| s.parse().ok()).unwrap_or(f64::NAN);
        let y = parts.next().and_then(|s| s.parse().ok()).unwrap_or(f64::NAN);
        return (x, y);
    }

    // Para qualquer outra geometria: extrair todos os números e calcular bbox centroid
    let coords: Vec<f64> = wkt
        .split(|c: char| !c.is_ascii_digit() && c != '-' && c != '.')
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();

    if coords.len() < 2 {
        return (f64::NAN, f64::NAN);
    }

    // Pares (x, y) interleaved
    let (mut min_x, mut max_x) = (f64::MAX, f64::MIN);
    let (mut min_y, mut max_y) = (f64::MAX, f64::MIN);
    let mut i = 0;
    while i + 1 < coords.len() {
        let cx = coords[i];
        let cy = coords[i + 1];
        if cx < min_x { min_x = cx; }
        if cx > max_x { max_x = cx; }
        if cy < min_y { min_y = cy; }
        if cy > max_y { max_y = cy; }
        i += 2;
    }
    ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[hayashi_fn] renomeia a fn original para __hayashi_impl_<nome>.

    #[test]
    fn test_morans_i() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![vec![0.0, 0.5, 0.5, 0.0, 0.0]; 5];
        let result = __hayashi_impl_morans_i(values, weights);
        assert!(result >= -1.0 && result <= 1.0);
    }

    #[test]
    fn test_distance_matrix() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 0.0, 0.0];
        let result = __hayashi_impl_distance_matrix(x, y);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0][1], 1.0);
    }

    #[test]
    fn test_spatial_lag() {
        let values = vec![1.0, 2.0, 3.0];
        let weights = vec![vec![0.0, 0.5, 0.5]; 3];
        let result = __hayashi_impl_spatial_lag(values, weights);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_inverse_distance_weights() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 0.0, 0.0];
        let result = __hayashi_impl_inverse_distance_weights(x, y, 1.0);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_wkt_to_coords_point() {
        let geoms = vec![
            Geometry::from_wkt("POINT(-43.5 -22.9)"),
            Geometry::from_wkt("POINT(-46.6 -23.5)"),
        ];
        let coords = __hayashi_impl_wkt_to_coords(geoms);
        let xs = &coords["x"];
        let ys = &coords["y"];
        assert!((xs[0] - (-43.5)).abs() < 1e-9);
        assert!((ys[0] - (-22.9)).abs() < 1e-9);
        assert!((xs[1] - (-46.6)).abs() < 1e-9);
        assert!((ys[1] - (-23.5)).abs() < 1e-9);
    }

    #[test]
    fn test_wkt_to_coords_polygon_centroid() {
        // Quadrado unitário centrado em (0.5, 0.5)
        let geoms = vec![Geometry::from_wkt(
            "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
        )];
        let coords = __hayashi_impl_wkt_to_coords(geoms);
        assert!((coords["x"][0] - 0.5).abs() < 1e-9);
        assert!((coords["y"][0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_parse_wkt_centroid_empty() {
        let (x, y) = parse_wkt_centroid("");
        assert!(x.is_nan());
        assert!(y.is_nan());
    }
}
