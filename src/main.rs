use burn::nn::{Linear, Module, MSELoss};
use burn::optim::{Adam, Optimizer};
use burn::tensor::backend::NdArray;
use burn::tensor::{Data, Tensor};
use rand::Rng;
use textplots::{Chart, Plot, Shape};

#[derive(Module, Debug)]
struct LinearRegressionModel {
    layer: Linear<NdArray, 1, 1>,
}

impl LinearRegressionModel {
    pub fn new() -> Self {
        Self {
            layer: Linear::new(),
        }
    }

    pub fn forward(&self, x: Tensor<NdArray, 1>) -> Tensor<NdArray, 1> {
        self.layer.forward(x)
    }
}

fn generate_data(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut x_vals = Vec::new();
    let mut y_vals = Vec::new();
    for _ in 0..n {
        let x: f32 = rng.gen_range(0.0..10.0);
        let noise: f32 = rng.gen_range(-1.0..1.0);
        let y = 2.0 * x + 1.0 + noise;
        x_vals.push(x);
        y_vals.push(y);
    }
    (x_vals, y_vals)
}

fn train(model: &mut LinearRegressionModel, x_train: &[f32], y_train: &[f32], epochs: usize) {
    let mut optimizer = Adam::new(model, 0.01);
    let loss_fn = MSELoss::new();

    for epoch in 0..epochs {
        let x_tensor = Tensor::<NdArray, 1>::from_data(Data::from(x_train.to_vec()));
        let y_tensor = Tensor::<NdArray, 1>::from_data(Data::from(y_train.to_vec()));
        let predictions = model.forward(x_tensor.clone());
        let loss = loss_fn.forward(predictions, y_tensor);
        optimizer.backward_step(model, loss.clone());

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss: {:.5}", epoch, loss.into_scalar());
        }
    }
}

fn plot_results(x_vals: &[f32], y_actual: &[f32], y_predicted: &[f32]) {
    println!("Visualizing Results:");
    let data: Vec<(f32, f32)> = x_vals.iter().zip(y_predicted).map(|(&x, &y)| (x, y)).collect();
    Chart::new(80, 25, 0.0, 10.0)
        .lineplot(&Shape::Lines(&data))
        .display();
}

fn main() {
    let (x_train, y_train) = generate_data(100);
    let mut model = LinearRegressionModel::new();
    train(&mut model, &x_train, &y_train, 100);

    let x_test: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let x_test_tensor = Tensor::<NdArray, 1>::from_data(Data::from(x_test.clone()));
    let y_predicted = model.forward(x_test_tensor).into_data().convert::<Vec<f32>>();
    plot_results(&x_test, &y_train[0..10], &y_predicted);
}
