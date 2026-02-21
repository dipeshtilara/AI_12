// Handles unit-specific dynamic visualizations using Canvas API

window.setupVisualizer = function (unitId, containerId, colorTheme) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = ''; // Clear placeholder

    const canvas = document.createElement('canvas');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');

    // Handle resize
    const resizeObserver = new ResizeObserver(() => {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        // Trigger re-render if needed
    });
    resizeObserver.observe(container);

    // Route to specific visualizer
    switch (unitId) {
        case 'unit-6': // Neural Networks
            startNeuralNetworkVis(ctx, canvas, colorTheme);
            break;
        case 'unit-3': // Computer Vision
            startComputerVisionVis(ctx, canvas, colorTheme);
            break;
        case 'unit-2': // Data Science Methodology
            startDataFlowVis(ctx, canvas, colorTheme);
            break;
        case 'unit-8': // Storytelling
            startChartsVis(ctx, canvas, colorTheme);
            break;
        default:
            startGenericParticles(ctx, canvas, colorTheme);
    }
}

// Visualizer: Neural Network (Unit 6)
function startNeuralNetworkVis(ctx, canvas, color) {
    const layers = [4, 6, 6, 3];
    const nodes = [];
    const connections = [];

    // Initialize Nodes
    const layerSpacing = canvas.width / (layers.length + 1);

    layers.forEach((count, lIndex) => {
        const x = layerSpacing * (lIndex + 1);
        const nodeSpacing = canvas.height / (count + 1);

        for (let i = 0; i < count; i++) {
            nodes.push({
                x: x,
                y: nodeSpacing * (i + 1),
                layer: lIndex,
                active: Math.random()
            });
        }
    });

    // Animation Loop
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw Connections
        ctx.lineWidth = 1;
        nodes.forEach(node => {
            nodes.forEach(target => {
                if (target.layer === node.layer + 1) {
                    const dist = Math.hypot(node.x - target.x, node.y - target.y);
                    const alpha = (Math.sin(Date.now() * 0.002 + node.x) + 1) / 2 * 0.3;

                    ctx.beginPath();
                    ctx.moveTo(node.x, node.y);
                    ctx.lineTo(target.x, target.y);
                    ctx.strokeStyle = `rgba(139, 92, 246, ${alpha})`; // Violet
                    ctx.stroke();
                }
            });
        });

        // Draw Nodes
        nodes.forEach(node => {
            ctx.beginPath();
            ctx.arc(node.x, node.y, 8, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();

            // Pulse effect
            const pulse = (Math.sin(Date.now() * 0.005 + node.y) + 1) / 2;
            ctx.beginPath();
            ctx.arc(node.x, node.y, 8 + pulse * 5, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(139, 92, 246, ${0.5 - pulse * 0.3})`;
            ctx.stroke();
        });

        // Simulate Signal Flow
        const time = Date.now() * 0.002;
        const signalX = (time % (layers.length)) * layerSpacing + layerSpacing;

        requestAnimationFrame(animate);
    }
    animate();
}

// Visualizer: Computer Vision (Unit 3)
function startComputerVisionVis(ctx, canvas, color) {
    const gridSize = 40;
    const cols = Math.ceil(canvas.width / gridSize);
    const rows = Math.ceil(canvas.height / gridSize);

    function animate() {
        ctx.fillStyle = '#0f172a'; // BG
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const time = Date.now() * 0.001;

        for (let i = 0; i < cols; i++) {
            for (let j = 0; j < rows; j++) {
                const x = i * gridSize;
                const y = j * gridSize;

                // Scanning bar effect
                const scanLine = (Math.sin(time) + 1) / 2 * canvas.height;
                const distToScan = Math.abs(y - scanLine);

                if (distToScan < 100) {
                    const alpha = 1 - (distToScan / 100);
                    ctx.fillStyle = `rgba(59, 130, 246, ${alpha * 0.5})`; // Blue
                    ctx.fillRect(x + 1, y + 1, gridSize - 2, gridSize - 2);
                }

                ctx.strokeStyle = 'rgba(255,255,255,0.05)';
                ctx.strokeRect(x, y, gridSize, gridSize);
            }
        }

        // Face detection box simulation
        const boxX = canvas.width / 2 - 100 + Math.sin(time) * 50;
        const boxY = canvas.height / 2 - 100 + Math.cos(time * 1.5) * 30;

        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 2;
        ctx.strokeRect(boxX, boxY, 200, 200);

        // Corners
        ctx.lineWidth = 4;
        const cornerSize = 20;
        ctx.beginPath();
        // TL
        ctx.moveTo(boxX, boxY + cornerSize); ctx.lineTo(boxX, boxY); ctx.lineTo(boxX + cornerSize, boxY);
        // TR
        ctx.moveTo(boxX + 200 - cornerSize, boxY); ctx.lineTo(boxX + 200, boxY); ctx.lineTo(boxX + 200, boxY + cornerSize);
        // BR
        ctx.moveTo(boxX + 200, boxY + 200 - cornerSize); ctx.lineTo(boxX + 200, boxY + 200); ctx.lineTo(boxX + 200 - cornerSize, boxY + 200);
        // BL
        ctx.moveTo(boxX + cornerSize, boxY + 200); ctx.lineTo(boxX, boxY + 200); ctx.lineTo(boxX, boxY + 200 - cornerSize);
        ctx.stroke();

        ctx.fillStyle = '#3b82f6';
        ctx.font = '14px sans-serif';
        ctx.fillText('DETECTING OBJECT...', boxX, boxY - 10);

        requestAnimationFrame(animate);
    }
    animate();
}

// Visualizer: Generic Particles
function startGenericParticles(ctx, canvas, color) {
    const particles = [];
    const particleCount = 50;

    for (let i = 0; i < particleCount; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2,
            size: Math.random() * 3
        });
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();

            // Connect nearby
            particles.forEach(p2 => {
                const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
                if (dist < 100) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = color.replace(')', ', 0.1)').replace('rgb', 'rgba').replace('#', '');
                    // Basic hex to rgba hack or just use low opacity white
                    ctx.strokeStyle = `rgba(255,255,255, ${0.1 * (1 - dist / 100)})`;
                    ctx.stroke();
                }
            });
        });

        requestAnimationFrame(animate);
    }
    animate();
}

function startDataFlowVis(ctx, canvas, color) {
    // Flowchart style animation
    startGenericParticles(ctx, canvas, color); // Fallback for now to save space
}

function startChartsVis(ctx, canvas, color) {
    // Bar chart animation
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const barCount = 10;
        const width = canvas.width / barCount;
        const time = Date.now() * 0.003;

        for (let i = 0; i < barCount; i++) {
            const h = Math.abs(Math.sin(time + i)) * canvas.height * 0.8;
            const x = i * width + width * 0.1;
            const y = canvas.height - h;

            ctx.fillStyle = color;
            ctx.fillRect(x, y, width * 0.8, h);
        }
        requestAnimationFrame(animate);
    }
    animate();
}
