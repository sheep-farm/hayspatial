#![allow(clippy::not_unsafe_ptr_arg_deref)]
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
    let numerator: f64 = values.iter().enumerate().map(|(i, &yi)| {
        weights.iter().take(n).map(|row| {
            if let Some(&wj) = row.get(i) {
                let yj = values.get(i).unwrap_or(&0.0);
                wj * (yj - mean) * (yi - mean)
            } else {
                0.0
            }
        }).sum::<f64>()
    }).sum();
    
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
    let numerator: f64 = values.iter().enumerate().map(|(i, &yi)| {
        weights.iter().take(n).map(|row| {
            if let Some(&wj) = row.get(i) {
                let yj = values.get(i).unwrap_or(&0.0);
                wj * (yj - yi).powi(2)
            } else {
                0.0
            }
        }).sum::<f64>()
    }).sum();
    
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
    
    values.iter().enumerate().map(|(i, _yi)| {
        weights.iter().take(n).map(|row| {
            if let Some(&wj) = row.get(i) {
                let yj = values.get(i).unwrap_or(&0.0);
                wj * yj
            } else {
                0.0
            }
        }).sum::<f64>()
    }).collect()
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
pub fn inverse_distance_weights(x_coords: Vec<f64>, y_coords: Vec<f64>, alpha: f64) -> Vec<Vec<f64>> {
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
    
    values.iter().enumerate().map(|(i, _yi)| {
        let local_sum: f64 = weights.iter().take(n).map(|row| {
            if let Some(&wj) = row.get(i) {
                let yj = values.get(i).unwrap_or(&0.0);
                wj * (yj - mean)
            } else {
                0.0
            }
        }).sum::<f64>();
        
        let yi = values.get(i).unwrap_or(&0.0);
        (yi - mean) / variance * local_sum
    }).collect()
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
    
    values.iter().enumerate().map(|(i, _yi)| {
        let local_sum: f64 = weights.iter().take(n).map(|row| {
            if let Some(&wj) = row.get(i) {
                let yj = values.get(i).unwrap_or(&0.0);
                wj * yj
            } else {
                0.0
            }
        }).sum::<f64>();
        
        let local_mean = local_sum / weights.iter().take(n).map(|row| row.iter().sum::<f64>()).sum::<f64>();
        
        (local_mean - mean) / (s2.sqrt() * ((weights.iter().take(n).map(|row| row.iter().sum::<f64>()).sum::<f64>() - 1.0) / (n - 1) as f64).sqrt())
    }).collect()
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
    let numerator: f64 = values.iter().skip(lag).enumerate().map(|(i, &yi)| {
        let yj = values[i];
        (yi - mean) * (yj - mean)
    }).sum();
    
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
        let mut distances: Vec<(usize, f64)> = (0..n).map(|j| {
            if i == j {
                (j, f64::INFINITY)
            } else {
                let dx = x_coords[i] - x_coords[j];
                let dy = y_coords[i] - y_coords[j];
                (j, (dx * dx + dy * dy).sqrt())
            }
        }).collect();
        
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
    (0..=max_lag).map(|lag| {
        let n = values.len();
        let lag = lag as usize;
        if n <= lag || weights.is_empty() {
            return 0.0;
        }
        
        let mean: f64 = values.iter().sum::<f64>() / n as f64;
        let numerator: f64 = values.iter().skip(lag).enumerate().map(|(i, &yi)| {
            let yj = values[i];
            (yi - mean) * (yj - mean)
        }).sum();
        
        let denominator: f64 = values.iter().map(|&y| (y - mean).powi(2)).sum();
        
        if denominator == 0.0 {
            return 0.0;
        }
        
        numerator / denominator
    }).collect()
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
}
