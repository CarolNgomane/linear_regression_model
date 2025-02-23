# Linear Regression

This project demonstrates how to implement a simple linear regression model using the **Burn** library in **Rust**. The goal was to predict values based on some noisy 
input data, visualize the results, and evaluate the model's performance. As a beginner in Rust, this was a great opportunity to explore machine learning concepts and 
dive deeper into the Rust ecosystem.

## 1. Introduction to the Problem

Linear regression is a fundamental machine learning algorithm used to model the relationship between a dependent variable (`y`) and an independent variable (`x`). 
For this project, I created a simple dataset where the output `y` is calculated using the formula:
\[ y = 2x + 1 + \text{noise} \]
where the noise is a random value added to make the data more realistic. The objective was to train a model to predict `y` from `x` and evaluate how well it fits the data.

## 2. Description of My Approach

Model Definition

I built the model using the **Burn** library in Rust. Since I’m still learning Rust, I encountered some challenges, but it helped me understand how to structure a machine 
learning model. I defined a model with a single **Linear** layer that takes input `x` and outputs a prediction for `y`.

```rust
#[derive(Module, Debug)]
struct LinearRegressionModel {
    layer: Linear<NdArray, 1, 1>,
}
impl LinearRegressionModel {
    pub fn new() -> Self {
        Self {
            layer: Linear::new(),
        } }
    pub fn forward(&self, x: Tensor<NdArray, 1>) -> Tensor<NdArray, 1> {
        self.layer.forward(x)
    }}


Data Generation
To simulate real-world data, I generated random values for x between 0 and 10. I then calculated the output y using the formula mentioned earlier, 
adding noise to the results to make the data less perfect. This dataset of 100 points was used to train the model.

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
Model Training
Trained the model using the Adam optimizer with a learning rate of 0.01, and I used MSE (Mean Squared Error) as the loss function to measure how close 
the model's predictions were to the actual values. The training was run for 100 epochs, and I printed the loss every 10 epochs to track the progress.

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
    )}
Visualization
After training the model, I tested it on a small set of test data and visualized the results using the Textplots library. This allowed me to generate simple ASCII charts in the terminal to compare the actual and predicted values.

fn plot_results(x_vals: &[f32], y_actual: &[f32], y_predicted: &[f32]) {
    println!("Visualizing Results:");
    let data: Vec<(f32, f32)> = x_vals.iter().zip(y_predicted).map(|(&x, &y)| (x, y)).collect();
    Chart::new(80, 25, 0.0, 10.0)
        .lineplot(&Shape::Lines(&data))
        .display();
}
3. Results and Evaluation of the Model
The model was able to train and reduce the loss over the 100 epochs, which is expected in a well-functioning linear regression model. 
Although the predictions were not perfect, the loss gradually decreased, indicating that the model was improving over time.

While the model didn’t achieve perfect predictions, it demonstrated how linear regression works and provided useful feedback on how the model was learning.

Visualization Challenges
I faced some challenges generating graphical visuals. While Textplots allowed me to create basic terminal plots, the charts were relatively simple and didn’t 
offer advanced functionality. If I were to improve this project, I’d explore graphical libraries or export the data to external tools for more sophisticated visualizations.


Challenges Encountered
Rust Syntax and Memory Management: As a beginner in Rust, understanding the syntax and the ownership system was challenging. Rust's memory management 
model (ownership, borrowing, and references) took time to grasp, especially when working with tensors and complex data structures.

Burn Library: The Burn library was helpful but also required a steep learning curve. There weren’t many examples, and the documentation was not always clear. 
I spent a lot of time experimenting to get the code working, which helped me learn more about Rust in the process.

Visualization: The visualization process was another challenge. Textplots is useful for simple, text-based charts, but it doesn’t provide the level of detail 
I would prefer. I was unable to generate detailed graphical plots, which limited my ability to fully evaluate the model’s performance.


Rust and Machine Learning: I learned a lot about how to structure a machine learning model in Rust. It was my first time using a machine learning library in Rust, 
and despite the challenges, I gained a better understanding of both the language and the basic principles of machine learning.

Model Training: I realized that model performance doesn’t always improve quickly, and hyperparameter tuning plays a significant role in making models more accurate. 
I also learned that regular evaluation during training helps track progress.

Rust’s Memory Model: Managing memory in Rust was difficult at first, but I’ve become more comfortable with the concepts of ownership and borrowing, 
which are essential for performance and safety in Rust applications.


AI Assistance: I used AI tools like ChatGPT & copilot to help clarify concepts, debug code, and suggest improvements. This greatly accelerated my learning 
process and helped me avoid common mistakes.
Tutorials from youtube to get some kind of introduction to rust.
Documentation: I relied on official documentation for both Rust and the Burn library. These resources were indispensable in understanding how to use the tools effectively.

