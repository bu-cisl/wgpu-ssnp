// Simulation control and validation
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
	const resVal = document.getElementById("resInput").value.trim() || "0.1,0.1,0.1";
	const naVal = document.getElementById("naInput").value.trim() || "0.65";
	const outputTypeVal = document.getElementById("outputType").value;
	const intensityVal = outputTypeVal === "intensity" ? "1" : "0";
	const n0Val = document.getElementById("n0Input").value.trim() || "1.33";
	
	if (!validateInputs(resVal, naVal, n0Val)) {
		return;
	}
	
	const angleString = `${currentAngle[0]},${currentAngle[1]}`;
	const combinedParams = `${angleString}|${resVal}|${naVal}|${intensityVal}|${n0Val}`;
	
	const viewer = document.getElementById("resultContainer");
	viewer.innerHTML = "";
	
	console.log("Sending input to C++");
	
	Module.ccall(
		"callSSNP",
		null,
		["string"],
		[combinedParams],
		{ async: true }
	).catch(err => console.error("Error:", err));
}

function initializeSimulation() {
	const runBtn = document.getElementById("runBtn");
	runBtn.addEventListener("click", runForwardFunction);
}