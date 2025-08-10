// Main application initialization
var Module = {
	noInitialRun: true,
	print: function (text) {
		console.log(text);
	},
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
	createAppStructure();
});

Module.onRuntimeInitialized = () => {
	// Initialize all modules
	initializeAngleSelector();
	initializeFileHandlers();
	initializeSimulation();
	
	// Initialize download functionality
	document.getElementById("downloadBtn").addEventListener("click", downloadNumPy);
};

// Memory cleanup
window.addEventListener('beforeunload', () => {
    if (window.currentVolumeData) {
        Module._free(window.currentVolumeData.ptr);
        window.currentVolumeData = null;
    }
});