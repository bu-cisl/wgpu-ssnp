// DOM structure builder
function createAppStructure() {
	document.body.innerHTML = `
		<h1>WebGPU SSNP Model Viewer</h1>
		<div class="author-link">
			<a href="https://github.com/andrewx-bu/wgpu_ssnp-idt" target="_blank" rel="noopener noreferrer">
				<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
					<path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
				</svg>
				Andrew Xin, Rayan Syed
			</a>
		</div>
		<div id="controls">
			<div class="control-group">
				<label for="fileInput">Upload Volume Data (.tiff)</label>
				<input type="file" id="fileInput" accept=".bin,.tif,.tiff" />
			</div>
			<div id="angleSelectorContainer">
				<label>Illumination Angle
				<span class="tooltip-icon">?
				<span class="tooltip-text">Illumination angle (ky, kx)/k0</span>
				</span>
				</label>
				<div id="angleSelector">
					<div id="angleIndicator"></div>
					<div id="anglePreview"></div>
				</div>
				<div id="currentAngle">Selected angle: (0.00, 0.00)</div>
			</div>
			<div class="control-group">
				<label for="resInput">
				Resolution
				<span class="tooltip-icon">?
				<span class="tooltip-text">Unitless measure of resolution (dz, dy, dx)/λ</span>
				</span>
				</label>
				<input type="text" id="resInput" placeholder="e.g.: 0.1,0.1,0.1" />
			</div>
			<div class="control-group">
				<label for="naInput">
				Numerical Aperture
				<span class="tooltip-icon">?
				<span class="tooltip-text">Numerical aperture of objective lens</span>
				</span>
				</label>
				<input type="text" id="naInput" placeholder="e.g.: 0.65" />
			</div>
			<div class="control-group">
				<label for="n0Input">
				Refractive Index
				<span class="tooltip-icon">?
				<span class="tooltip-text">Background refractive index</span>
				</span>
				</label>
				<input type="text" id="n0Input" placeholder="e.g.: 1.33" />
			</div>
			<div class="control-group">
				<label for="outputType">Output Type</label>
				<select id="outputType">
					<option value="amplitude">Amplitude</option>
					<option value="intensity" selected>Intensity</option>
					<option value="complex">Complex</option>
				</select>
			</div>
			<button id="runBtn" disabled>Run Simulation</button>
			<button id="downloadBtn" disabled style="margin-top: 10px;">Download as .npy</button>
		</div>
		<div id="viewer">
			<div class="result-container" id="resultContainer"></div>
		</div>
	`;
}