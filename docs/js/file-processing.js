// File processing utilities
function getSampleFormatName(format) {
	return ['undefined', 'unsigned integer', 'signed integer', 'IEEE float'][format] || `unknown (${format})`;
}

function getPhotometricName(photometric) {
	const names = {
		0: 'WhiteIsZero',
		1: 'BlackIsZero',
		2: 'RGB',
		3: 'RGB Palette',
		4: 'Transparency Mask',
		5: 'CMYK',
		6: 'YCbCr',
		8: 'CIELab'
	};
	return names[photometric] || `unknown (${photometric})`;
}

function convertTiffDataToFloat32(raw, metadata) {
	const { bitsPerSample, samplesPerPixel, sampleFormat, width, height } = metadata;
	const expectedLength = width * height * samplesPerPixel;
	
	if (raw instanceof Float32Array) return raw;
	if (raw instanceof Float64Array) return new Float32Array(raw);

	const output = new Float32Array(expectedLength);
	
	switch (sampleFormat) {
		case 1: // Unsigned integer
			if (bitsPerSample === 8) {
				for (let i = 0; i < raw.length; i++) {
					output[i] = raw[i] / 0xFF;
				}
			} else if (bitsPerSample === 16) {
				const uint16 = raw instanceof Uint16Array ? raw : new Uint16Array(raw.buffer);
				for (let i = 0; i < uint16.length; i++) {
					output[i] = uint16[i] / 0xFFFF;
				}
			}
			break;
			
		case 2: // Signed integer
			if (bitsPerSample === 8) {
				const int8 = raw instanceof Int8Array ? raw : new Int8Array(raw.buffer);
				for (let i = 0; i < int8.length; i++) {
					output[i] = int8[i] / 128.0;
				}
			} else if (bitsPerSample === 16) {
				const int16 = raw instanceof Int16Array ? raw : new Int16Array(raw.buffer);
				for (let i = 0; i < int16.length; i++) {
					output[i] = int16[i] / 32768.0;
				}
			}
			break;
			
		case 3: // Float
			if (bitsPerSample === 32) {
				return raw instanceof Float32Array ? raw : new Float32Array(raw.buffer);
			} else if (bitsPerSample === 64) {
				const float64 = raw instanceof Float64Array ? raw : new Float64Array(raw.buffer);
				for (let i = 0; i < float64.length; i++) {
					output[i] = float64[i];
				}
			}
			break;
			
		default:
			console.warn("Unknown sample format. Raw values will not be normalized.");
			for (let i = 0; i < raw.length; i++) {
				output[i] = raw[i];
			}
	}
	
	return output;
}

function decodeTiffToRaw(arrayBuffer) {
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

	return { slices, width, height, depth: slices.length };
}