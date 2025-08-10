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