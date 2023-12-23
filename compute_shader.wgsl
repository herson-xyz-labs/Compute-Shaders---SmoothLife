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

fn applyRulesToCell(innerKernelStateNormalized: f32, outerKernelStateNormalized: f32) -> f32 {
    // // // Original Values
    // var B1: f32 = 0.278;
    // var B2: f32 = 0.365;
    // var D1: f32 = 0.267;
    // var D2: f32 = 0.445;
    // var M: f32 = 0.147;
    // var N: f32 = 0.028;

    // // Version 2
    // var B1: f32 = 0.278;
    // var B2: f32 = 0.365;
    // var D1: f32 = 0.276;
    // var D2: f32 = 0.445;
    // var M: f32 = 0.147;
    // var N: f32 = 0.028;

    // Version 3
    var B1: f32 = 0.278;
    var B2: f32 = 0.365;
    var D1: f32 = 0.287;
    var D2: f32 = 0.445;
    var M: f32 = 0.147;
    var N: f32 = 0.1;

    var aliveness: f32 = logistic_threshold(innerKernelStateNormalized, 0.005, M);
    var threshold1: f32 = lerp(B1, D1, aliveness);
    var threshold2: f32 = lerp(B2, D2, aliveness);
    var new_aliveness: f32 = logistic_interval(outerKernelStateNormalized, threshold1, threshold2, N);

    // Clip new aliveness to be between 0 and 1
    var newCellState: f32 = clamp(new_aliveness, 0.0, 1.0);

    return newCellState;
}

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn computeMain(
        @builtin(global_invocation_id) cell : vec3u,
        @builtin(local_invocation_id) local_id : vec3<u32>,
    ) {
        var innerRadius: f32 = 2.0;
        var outerRadius: f32 = innerRadius * 3.0;

        // Current state of the cell
        var kernelResults: KernelResults = processCell(cell.xy, innerRadius, outerRadius);
        // New state of the cell
        var newCellState: f32 = applyRulesToCell(kernelResults.innerKernelStateNormalized, kernelResults.outerKernelStateNormalized);

        let cellIndex = getCellIndex(cell.xy);
        let cellState = getCellState(cell.x, cell.y);

        cellStateOut[cellIndex] = newCellState;

        var cellInfo: CellInfo = CellInfo(cell.xy, 0);

        cellInfo = CellInfo(
                            cell.xy, 
                            kernelResults.innerKernelStateNormalized, 
        );

        output[getCellIndex(cell.xy)] = cellInfo;
}  