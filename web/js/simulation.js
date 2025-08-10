// Simulation control and validation
let isSimulationRunning = false;

function resetSimulationState() {
	isSimulationRunning = false;
	const runBtn = document.getElementById("runBtn");
	const angleSelector = document.getElementById("angleSelector");
	runBtn.disabled = false;
	runBtn.textContent = "Run Simulation";
	angleSelector.style.pointerEvents = "auto";
	angleSelector.style.opacity = "1";
}

function validateInputs(resVal, naVal, n0Val) {
	const resParts = resVal.split(',');
	if (resParts.length !== 3 || resParts.some(part => isNaN(parseFloat(part)))) {
		alert("Resolution must be exactly 3 numbers separated by commas (e.g. '0.1,0.1,0.1')");
		return false;
	}
	
	if (isNaN(naVal) || isNaN(parseFloat(naVal))) {
		alert("NA must be a single number (e.g. '0.65')");
		return false;
	}
	
	if (isNaN(n0Val) || isNaN(parseFloat(n0Val))) {
		alert("n0 must be a single number (e.g. '1.33')");
		return false;
	}
	
	return true;
}

function runForwardFunction() {
	if (!window.currentVolumeData) {
        alert("No volume data loaded");
        return;
    }
	
	const resVal = document.getElementById("resInput").value.trim() || "0.1,0.1,0.1";
	const naVal = document.getElementById("naInput").value.trim() || "0.65";
	const outputTypeVal = document.getElementById("outputType").value;
	let outputTypeNumVal;
	if (outputTypeVal === "intensity") {
		outputTypeNumVal = "1";
	} else if (outputTypeVal === "complex") {
		outputTypeNumVal = "2";
	} else {
		outputTypeNumVal = "0"; // amplitude
	}
	const n0Val = document.getElementById("n0Input").value.trim() || "1.33";
	
	if (!validateInputs(resVal, naVal, n0Val)) {
		return;
	}
	
	// Set running state and disable controls
	isSimulationRunning = true;
	const runBtn = document.getElementById("runBtn");
	const angleSelector = document.getElementById("angleSelector");
	runBtn.disabled = true;
	runBtn.textContent = "Running...";
	angleSelector.style.pointerEvents = "none";
	angleSelector.style.opacity = "0.5";
	
	const { ptr, D, H, W } = window.currentVolumeData;
	const angleString = `${currentAngle[0]},${currentAngle[1]}`;
	const combinedParams = `${angleString}|${resVal}|${naVal}|${outputTypeNumVal}|${n0Val}`;
	
	const viewer = document.getElementById("resultContainer");
	viewer.innerHTML = "";
	
	console.log("Sending input to C++");
	
	Module.ccall(
        "callSSNP", 
        null,
        ["number", "number", "number", "number", "string"],
        [ptr, D, H, W, combinedParams],
        { async: true }
    ).catch(err => {
		console.error("Error:", err);
		// Re-enable controls on error
		resetSimulationState();
	});
}

function initializeSimulation() {
	const runBtn = document.getElementById("runBtn");
	runBtn.addEventListener("click", runForwardFunction);
}