// File upload and processing handlers
function initializeFileHandlers() {
	const fileInput = document.getElementById("fileInput");
	const runBtn = document.getElementById("runBtn");

	fileInput.addEventListener("change", async (e) => {
		// Clean up previous data 
		if (window.currentVolumeData) {
			Module._free(window.currentVolumeData.ptr);
			window.currentVolumeData = null;
		}

		const file = e.target.files[0];
		if (!file) return;

		const name = file.name.toLowerCase();
		const ext = name.substring(name.lastIndexOf(".") + 1);

		if (ext === "tif" || ext === "tiff") {
			try {
				const arrayBuffer = await file.arrayBuffer();
				const { slices, width, height, depth } = decodeTiffToRaw(arrayBuffer);

				// Allocate memory on heap and copy data
				const D = depth;
				const H = height; 
				const W = width;
				const totalFloats = D * H * W;

				const heapPtr = Module._malloc(totalFloats * 4);
				const heapArray = new Float32Array(Module.HEAPF32.buffer, heapPtr, totalFloats);

				// Copy slice data directly
				let idx = 0;
				for (let d = 0; d < D; d++) {
					for (let i = 0; i < slices[d].length; i++) {
						heapArray[idx++] = slices[d][i];
					}
				}

				// Store for cleanup
				window.currentVolumeData = { ptr: heapPtr, D, H, W };
				runBtn.disabled = false;

			} catch (error) {
				console.error("TIFF processing error:", error);
				alert(`Error processing TIFF: ${error.message}`);
			}
		} 
		else {
			alert("Unsupported file type: please select a .tif/.tiff");
		}
	});
}