use akson::ContinousFiniteLTISystem;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;
use tch::Tensor;

fn bench_discretize(c: &mut Criterion) {
    let system = black_box(
        ContinousFiniteLTISystem::new(
            Tensor::rand([10, 10], (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::rand([10, 1], (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::rand([1, 10], (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::rand([1, 1], (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::rand([10, 1], (tch::Kind::Double, tch::Device::Cpu)),
        )
        .unwrap(),
    );
    let t = black_box(0.1);

    c.bench_function("discretize", |b| b.iter(|| system.discretize(t)));
}

criterion_group! {
    name = benches;
    config = Criterion::default()
            .sample_size(500)
            .warm_up_time(Duration::from_secs(5))
            .measurement_time(Duration::from_secs(5));
    targets = bench_discretize
}
criterion_main!(benches);
