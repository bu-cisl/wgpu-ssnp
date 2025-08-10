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
				const ifds = UTIF.decode(arrayBuffer);
				
				if (ifds.length === 0) {
					throw new Error("No image slices found in TIFF");
				}

				UTIF.decodeImage(arrayBuffer, ifds[0]);
				const firstIfd = ifds[0];
				
				const width = firstIfd.width || firstIfd['t256']?.[0];
				const height = firstIfd.height || firstIfd['t257']?.[0];
				const bps = firstIfd.bps || firstIfd['t258']?.[0] || 8;
				const spp = firstIfd.spp || firstIfd['t277']?.[0] || 1;
				const sampleFormat = firstIfd.sampleFormat || firstIfd['t339']?.[0] || 1;
				const photometric = firstIfd.photometric || firstIfd['t262']?.[0];
				
				if (!width || !height) {
					throw new Error("Could not determine image dimensions");
				}

				console.log("TIFF Metadata:", {
					width,
					height,
					bitsPerSample: bps,
					samplesPerPixel: spp,
					sampleFormat: getSampleFormatName(sampleFormat),
					photometricInterpretation: getPhotometricName(photometric),
					compression: firstIfd.compression,
					planarConfiguration: firstIfd.planarConfig,
					ifdEntries: Object.keys(firstIfd).filter(k => k.startsWith('t'))
				});

				const slices = [];
				for (let i = 0; i < ifds.length; i++) {
					UTIF.decodeImage(arrayBuffer, ifds[i]);
					const ifd = ifds[i];
					const raw = ifd.data;
					
					if (!raw) {
						throw new Error(`Could not decode slice #${i}`);
					}

					const sliceWidth = ifd.width || ifd['t256']?.[0];
					const sliceHeight = ifd.height || ifd['t257']?.[0];
					if (sliceWidth !== width || sliceHeight !== height) {
						throw new Error(`Slice ${i} dimensions (${sliceWidth}x${sliceHeight}) don't match first slice (${width}x${height})`);
					}

					const floatSlice = convertTiffDataToFloat32(raw, {
						bitsPerSample: bps,
						samplesPerPixel: spp,
						sampleFormat,
						width,
						height
					});

					if (spp > 1) {
						const singleChannel = new Float32Array(width * height);
						for (let p = 0; p < width * height; p++) {
							singleChannel[p] = floatSlice[p * spp];
						}
						slices.push(singleChannel);
					} else {
						slices.push(floatSlice);
					}
				}

				// Allocate memory on heap and copy data
				const D = slices.length;
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