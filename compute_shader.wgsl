@group(0) @binding(0) var<uniform> grid: vec2f;
@group(0) @binding(1) var<storage> cellStateIn: array<f32>;
@group(0) @binding(2) var<storage, read_write> cellStateOut: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<CellInfo>;

struct CellInfo {
    position: vec2u,
    newCellState: f32,
}

struct KernelResults {
    innerKernelCellTotal: f32,
    innerKernelStateSum: f32,
    innerKernelStateNormalized: f32,
    outerKernelCellTotal: f32,
    outerKernelStateSum: f32,
    outerKernelStateNormalized: f32
};

fn getCellIndex(cell: vec2u) -> u32 {
    return (cell.y % u32(grid.y)) * u32(grid.x) +
        (cell.x % u32(grid.x));
}

fn getCellState(x: u32, y: u32) -> f32 {
    return cellStateIn[getCellIndex(vec2(x, y))];
}

fn sigmoid_a(x: f32, a: f32, b: f32) -> f32 {
    return 1.0 / (1.0 + exp(-(x - a) * 4 / b));
}

fn sigmoid_b(x: f32, b: f32, eb: f32) -> f32 {
    return 1.0 - sigmoid_a(x, b, eb);
}

fn sigmoid_ab(x: f32, a: f32, b: f32, ea: f32, eb: f32) -> f32 {
    return sigmoid_a(x, a, ea) * sigmoid_b(x, b, eb);
}

fn sigmoid_mix(x: f32, y: f32, m: f32, em: f32) -> f32 {
    return x * (1.0 - sigmoid_a(m, 0.5, em)) + y * sigmoid_a(m, 0.5, em);
}

fn logistic_threshold(x: f32, x0: f32, alpha: f32) -> f32 {
    return 1.0 / (1.0 + exp(-4.0 / alpha * (x - x0)));
}

fn logistic_interval(x: f32, a: f32, b: f32, alpha: f32) -> f32 {
    return logistic_threshold(x, a, alpha) * (1.0 - logistic_threshold(x, b, alpha));
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return (1.0 - t) * a + t * b;
}

// Function to calculate distance between two cells
fn calculateDistance(cell: vec2f, neighborCell: vec2f) -> f32 {
    let dx: f32 = neighborCell.x - cell.x;
    let dy: f32 = neighborCell.y - cell.y;
    return sqrt((dx * dx) + (dy * dy));
}

fn processCell(cell: vec2u, innerRadius: f32, outerRadius: f32) -> KernelResults {
    var results: KernelResults;
    results.innerKernelCellTotal = 0.0;
    results.innerKernelStateSum = 0.0;
    results.outerKernelCellTotal = 0.0;
    results.outerKernelStateSum = 0.0;

    let r: i32 = i32(outerRadius);

    for (var x = -r; x <= r; x++) {
        for (var y = -r; y <= r; y++) {
            let neighborCell: vec2f = vec2f(cell.xy) + vec2f(f32(x), f32(y));
            let dist_from_center: f32 = calculateDistance(vec2f(cell.xy), neighborCell);
            let logres: f32 = 4.0;
            let weight: f32 = 1.0 / (1.0 + exp(logres * (dist_from_center - outerRadius)));
            let neighborState = getCellState(cell.x + u32(x), cell.y + u32(y));

            if (dist_from_center < innerRadius) {
                results.innerKernelCellTotal += weight;
                results.innerKernelStateSum += (weight * neighborState);
            } else if (dist_from_center > innerRadius && dist_from_center <= outerRadius) {
                results.outerKernelCellTotal += weight;
                results.outerKernelStateSum += (weight * neighborState);
            }
        }
    }

    results.innerKernelStateNormalized = results.innerKernelStateSum / results.innerKernelCellTotal;
    results.outerKernelStateNormalized = results.outerKernelStateSum / results.outerKernelCellTotal;
    
    return results;
}

fn sigmoidInterval(x: f32, a: f32, b: f32, alpha: f32) -> f32 {
    return sigmoid_a(x, alpha, b) * (1.0 - sigmoid_a(x, alpha, b));
}

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn computeMain(
        @builtin(global_invocation_id) cell : vec3u,
        @builtin(local_invocation_id) local_id : vec3<u32>,
    ) {
        var innerRadius: f32 = 7.0;
        var outerRadius: f32 = innerRadius * 3.0;
        var kernelResults: KernelResults = processCell(cell.xy, innerRadius, outerRadius);

        var b1: f32 = 0.25;
        var b2: f32 = 0.36;
        var d1: f32 = 0.36;
        var d2: f32 = 0.46;
        var alpha_m: f32 = 0.147;
        var alpha_n: f32 = 0.028;
        var newCellState: f32 = 0;

        let cellIndex = getCellIndex(cell.xy);
        let cellState = getCellState(cell.x, cell.y);

        var aliveness_a: f32 = sigmoid_a(kernelResults.innerKernelStateNormalized, 0.5, alpha_m);
        var aliveness_b: f32 = sigmoid_b(kernelResults.outerKernelStateNormalized, 0.5, alpha_n);
        var aliveness_c: f32 = sigmoid_ab(kernelResults.innerKernelStateNormalized, b1, b2, alpha_m, alpha_m);
        var aliveness_d: f32 = sigmoid_mix(aliveness_a, aliveness_b, kernelResults.outerKernelStateNormalized, alpha_m);

        var aliveness_e: f32 = sigmoid_mix(sigmoid_ab(kernelResults.innerKernelStateNormalized, b1, b2, alpha_m, alpha_n),
                                        sigmoid_ab(kernelResults.innerKernelStateNormalized, d1, d2, alpha_m, alpha_n), 
                                        kernelResults.outerKernelStateNormalized, 
                                        alpha_m
                                        );
                                        
        cellStateOut[cellIndex] = aliveness_d;

        // Initialize the cell info.
        var cellInfo: CellInfo = CellInfo(cell.xy, 0);

        cellInfo = CellInfo(
                            cell.xy, 
                            kernelResults.innerKernelStateNormalized, 
        );

        output[getCellIndex(cell.xy)] = cellInfo;
}  