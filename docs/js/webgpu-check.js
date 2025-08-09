// WebGPU compatibility check
if (!navigator.gpu) {
	alert("Your browser does not support WebGPU. Please use a compatible browser such as the latest version of Google Chrome or Edge.");
	document.addEventListener("DOMContentLoaded", () => {
		document.getElementById("fileInput").disabled = true;
		document.getElementById("runBtn").disabled = true;
	});
}