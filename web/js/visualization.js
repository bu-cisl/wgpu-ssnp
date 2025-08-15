// Visualization utilities
let currentResultData = null;

function colormap(t) {
	t = Math.max(0, Math.min(1, t));
	const c = d3.color(d3.interpolateViridis(t));
	return [c.r, c.g, c.b];
}

function plotSlices(flatArray, H, W, localMin, localMax) {
	currentResultData = { data: flatArray, H, W };
	currentComplexData = null; // Reset complex data

	const resultContainer = document.getElementById('resultContainer');
	resultContainer.innerHTML = '';

	const DISPLAY_SIZE = 512;
	const scale = localMax - localMin || 1;

	const container = document.createElement('div');
	container.className = 'result-display';

	const canvas = document.createElement('canvas');
	canvas.width = DISPLAY_SIZE;
	canvas.height = DISPLAY_SIZE;
	const ctx = canvas.getContext('2d');

	const tempCanvas = document.createElement('canvas');
	tempCanvas.width = W;
	tempCanvas.height = H;
	const tempCtx = tempCanvas.getContext('2d');
	const tempImgData = tempCtx.createImageData(W, H);

	let idx = 0;
	for (let i = 0; i < H; i++) {
		for (let j = 0; j < W; j++) {
			const value = flatArray[idx++];
			const norm = (value - localMin) / scale;
			const [r, g, b] = colormap(norm);
			const pixelIndex = (i * W + j) * 4;
			tempImgData.data[pixelIndex] = r;
			tempImgData.data[pixelIndex + 1] = g;
			tempImgData.data[pixelIndex + 2] = b;
			tempImgData.data[pixelIndex + 3] = 255;
		}
	}

	tempCtx.putImageData(tempImgData, 0, 0);

	const scaleFactor = Math.min(DISPLAY_SIZE / W, DISPLAY_SIZE / H);
	const scaledWidth = W * scaleFactor;
	const scaledHeight = H * scaleFactor;
	const offsetX = (DISPLAY_SIZE - scaledWidth) / 2;
	const offsetY = (DISPLAY_SIZE - scaledHeight) / 2;

	ctx.imageSmoothingEnabled = false;
	ctx.clearRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
	ctx.drawImage(tempCanvas, 0, 0, W, H, offsetX, offsetY, scaledWidth, scaledHeight);

	container.appendChild(canvas);

	// Colorbar
	const colorbar = document.createElement('div');
	colorbar.className = 'colorbar';
	const colorCanvas = document.createElement('canvas');
	colorCanvas.width = 20;
	colorCanvas.height = DISPLAY_SIZE;
	const colorCtx = colorCanvas.getContext('2d');
	const colorImg = colorCtx.createImageData(1, DISPLAY_SIZE);

	for (let i = 0; i < DISPLAY_SIZE; i++) {
		const t = 1 - i / (DISPLAY_SIZE - 1);
		const [r, g, b] = colormap(t);
		const idx2 = i * 4;
		colorImg.data[idx2] = r;
		colorImg.data[idx2 + 1] = g;
		colorImg.data[idx2 + 2] = b;
		colorImg.data[idx2 + 3] = 255;
	}
	colorCtx.putImageData(colorImg, 0, 0);
	colorCtx.drawImage(colorCanvas, 0, 0, 1, DISPLAY_SIZE, 0, 0, 20, DISPLAY_SIZE);

	colorbar.appendChild(colorCanvas);
	const maxLabel = document.createElement('div');
	maxLabel.className = 'colorbar-label';
	maxLabel.textContent = localMax.toExponential(3);
	const minLabel = document.createElement('div');
	minLabel.className = 'colorbar-label';
	minLabel.textContent = localMin.toExponential(3);
	colorbar.insertBefore(maxLabel, colorCanvas);
	colorbar.appendChild(minLabel);

	container.appendChild(colorbar);
	resultContainer.appendChild(container);

	// Re-enable controls
	resetSimulationState();

	document.getElementById("downloadBtn").disabled = false;
}

function downloadNumPy() {
	if (currentComplexData) {
		downloadComplexNumPy();
	} else if (currentResultData) {
		const { data, H, W } = currentResultData;

		const headerDict = `{'descr': '<f4', 'fortran_order': False, 'shape': (${H}, ${W},)}`;
		const encoder = new TextEncoder();
		const headerBytes = encoder.encode(headerDict);

		const padLen = (16 - ((10 + headerBytes.length + 1) % 16)) % 16;

		const headerBlock = new Uint8Array(headerBytes.length + padLen + 1);
		headerBlock.set(headerBytes, 0);
		for (let i = headerBytes.length; i < headerBytes.length + padLen; i++) {
			headerBlock[i] = 0x20;
		}
		headerBlock[headerBlock.length - 1] = 0x0A;

		const totalLen = 10 + headerBlock.length + data.length * 4;
		const buffer = new Uint8Array(totalLen);

		buffer.set([0x93,0x4E,0x55,0x4D,0x50,0x59, 1, 0], 0);
		buffer[8] = headerBlock.length & 0xFF;
		buffer[9] = (headerBlock.length >> 8) & 0xFF;

		buffer.set(headerBlock, 10);

		new Float32Array(buffer.buffer, 10 + headerBlock.length).set(data);

		const blob = new Blob([buffer], { type: 'application/octet-stream' });
		const a = document.createElement('a');
		a.href = URL.createObjectURL(blob);
		a.download = 'idt_output.npy';
		a.click();
		URL.revokeObjectURL(a.href);
	}
}

let currentComplexData = null;

function plotComplexSlices(complexArray, H, W, magMin, magMax, phaseMin, phaseMax) {
	currentComplexData = { data: complexArray, H, W };
	currentResultData = null; // Reset regular data

	const resultContainer = document.getElementById('resultContainer');
	resultContainer.innerHTML = '';

	// Create magnitude display
	const magTitle = document.createElement('h3');
	magTitle.textContent = 'Magnitude';
	magTitle.style.textAlign = 'center';
	magTitle.style.margin = '20px 0 10px 0';
	resultContainer.appendChild(magTitle);

	const magContainer = createComplexPlot(complexArray, H, W, magMin, magMax, true);
	resultContainer.appendChild(magContainer);

	// Create phase display
	const phaseTitle = document.createElement('h3');
	phaseTitle.textContent = 'Phase';
	phaseTitle.style.textAlign = 'center';
	phaseTitle.style.margin = '20px 0 10px 0';
	resultContainer.appendChild(phaseTitle);

	const phaseContainer = createComplexPlot(complexArray, H, W, phaseMin, phaseMax, false);
	resultContainer.appendChild(phaseContainer);

	// Re-enable controls
	resetSimulationState();

	document.getElementById("downloadBtn").disabled = false;
}

function createComplexPlot(complexArray, H, W, localMin, localMax, isMagnitude) {
	const DISPLAY_SIZE = 512;
	const scale = localMax - localMin || 1;

	const container = document.createElement('div');
	container.className = 'result-display';

	const canvas = document.createElement('canvas');
	canvas.width = DISPLAY_SIZE;
	canvas.height = DISPLAY_SIZE;
	const ctx = canvas.getContext('2d');

	const tempCanvas = document.createElement('canvas');
	tempCanvas.width = W;
	tempCanvas.height = H;
	const tempCtx = tempCanvas.getContext('2d');
	const tempImgData = tempCtx.createImageData(W, H);

	for (let i = 0; i < H; i++) {
		for (let j = 0; j < W; j++) {
			const idx = (i * W + j) * 2;
			const real = complexArray[idx];
			const imag = complexArray[idx + 1];
			
			let value;
			if (isMagnitude) {
				value = Math.sqrt(real * real + imag * imag);
			} else {
				value = Math.atan2(imag, real);
			}
			
			const norm = (value - localMin) / scale;
			const [r, g, b] = colormap(norm);
			const pixelIndex = (i * W + j) * 4;
			tempImgData.data[pixelIndex] = r;
			tempImgData.data[pixelIndex + 1] = g;
			tempImgData.data[pixelIndex + 2] = b;
			tempImgData.data[pixelIndex + 3] = 255;
		}
	}

	tempCtx.putImageData(tempImgData, 0, 0);

	const scaleFactor = Math.min(DISPLAY_SIZE / W, DISPLAY_SIZE / H);
	const scaledWidth = W * scaleFactor;
	const scaledHeight = H * scaleFactor;
	const offsetX = (DISPLAY_SIZE - scaledWidth) / 2;
	const offsetY = (DISPLAY_SIZE - scaledHeight) / 2;

	ctx.imageSmoothingEnabled = false;
	ctx.clearRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
	ctx.drawImage(tempCanvas, 0, 0, W, H, offsetX, offsetY, scaledWidth, scaledHeight);

	container.appendChild(canvas);

	// Colorbar
	const colorbar = document.createElement('div');
	colorbar.className = 'colorbar';
	const colorCanvas = document.createElement('canvas');
	colorCanvas.width = 20;
	colorCanvas.height = DISPLAY_SIZE;
	const colorCtx = colorCanvas.getContext('2d');
	const colorImg = colorCtx.createImageData(1, DISPLAY_SIZE);

	for (let i = 0; i < DISPLAY_SIZE; i++) {
		const t = 1 - i / (DISPLAY_SIZE - 1);
		const [r, g, b] = colormap(t);
		const idx2 = i * 4;
		colorImg.data[idx2] = r;
		colorImg.data[idx2 + 1] = g;
		colorImg.data[idx2 + 2] = b;
		colorImg.data[idx2 + 3] = 255;
	}
	colorCtx.putImageData(colorImg, 0, 0);
	colorCtx.drawImage(colorCanvas, 0, 0, 1, DISPLAY_SIZE, 0, 0, 20, DISPLAY_SIZE);

	colorbar.appendChild(colorCanvas);
	const maxLabel = document.createElement('div');
	maxLabel.className = 'colorbar-label';
	maxLabel.textContent = localMax.toExponential(3);
	const minLabel = document.createElement('div');
	minLabel.className = 'colorbar-label';
	minLabel.textContent = localMin.toExponential(3);
	colorbar.insertBefore(maxLabel, colorCanvas);
	colorbar.appendChild(minLabel);

	container.appendChild(colorbar);
	return container;
}

function downloadComplexNumPy() {
	if (!currentComplexData) return;
	const { data, H, W } = currentComplexData;

	const headerDict = `{'descr': '<c8', 'fortran_order': False, 'shape': (${H}, ${W},)}`;
	const encoder = new TextEncoder();
	const headerBytes = encoder.encode(headerDict);

	const padLen = (16 - ((10 + headerBytes.length + 1) % 16)) % 16;

	const headerBlock = new Uint8Array(headerBytes.length + padLen + 1);
	headerBlock.set(headerBytes, 0);
	for (let i = headerBytes.length; i < headerBytes.length + padLen; i++) {
		headerBlock[i] = 0x20;
	}
	headerBlock[headerBlock.length - 1] = 0x0A;

	const totalLen = 10 + headerBlock.length + data.length * 4;
	const buffer = new Uint8Array(totalLen);

	buffer.set([0x93,0x4E,0x55,0x4D,0x50,0x59, 1, 0], 0);
	buffer[8] = headerBlock.length & 0xFF;
	buffer[9] = (headerBlock.length >> 8) & 0xFF;

	buffer.set(headerBlock, 10);

	new Float32Array(buffer.buffer, 10 + headerBlock.length).set(data);

	const blob = new Blob([buffer], { type: 'application/octet-stream' });
	const a = document.createElement('a');
	a.href = URL.createObjectURL(blob);
	a.download = 'idt_complex_output.npy';
	a.click();
	URL.revokeObjectURL(a.href);
}