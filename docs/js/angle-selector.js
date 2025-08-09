// Angle selector functionality
let currentAngle = [0, 0];

function initializeAngleSelector() {
	const angleSelector = document.getElementById("angleSelector");
	const anglePreview = document.getElementById("anglePreview");
	const angleIndicator = document.getElementById("angleIndicator");
	const currentAngleDisplay = document.getElementById("currentAngle");

	angleSelector.addEventListener("mousemove", (e) => {
		const rect = angleSelector.getBoundingClientRect();
		const centerX = rect.width / 2;
		const centerY = rect.height / 2;
		
		const mouseX = e.clientX - rect.left - centerX;
		const mouseY = -(e.clientY - rect.top - centerY);
		
		const distance = Math.sqrt(mouseX * mouseX + mouseY * mouseY);
		const maxDistance = centerX;
		
		let normalizedX = mouseX / maxDistance;
		let normalizedY = mouseY / maxDistance;
		
		if (distance > maxDistance) {
			const angle = Math.atan2(mouseY, mouseX);
			normalizedX = Math.cos(angle);
			normalizedY = Math.sin(angle);
		}
		
		anglePreview.style.display = "block";
		anglePreview.style.left = `${e.clientX - rect.left + 10}px`;
		anglePreview.style.top = `${e.clientY - rect.top + 10}px`;
		anglePreview.textContent = `(${normalizedX.toFixed(2)}, ${normalizedY.toFixed(2)})`;
	});

	angleSelector.addEventListener("mouseleave", () => {
		anglePreview.style.display = "none";
	});

	angleSelector.addEventListener("click", (e) => {
		const rect = angleSelector.getBoundingClientRect();
		const centerX = rect.width / 2;
		const centerY = rect.height / 2;
		
		const mouseX = e.clientX - rect.left - centerX;
		const mouseY = -(e.clientY - rect.top - centerY);
		
		const distance = Math.sqrt(mouseX * mouseX + mouseY * mouseY);
		const maxDistance = centerX;
		
		let normalizedX = mouseX / maxDistance;
		let normalizedY = mouseY / maxDistance;
		
		if (distance > maxDistance) {
			const angle = Math.atan2(mouseY, mouseX);
			normalizedX = Math.cos(angle);
			normalizedY = Math.sin(angle);
		}
		
		currentAngle = [normalizedX, normalizedY];
		currentAngleDisplay.textContent = `Selected angle: (${normalizedX.toFixed(2)}, ${normalizedY.toFixed(2)})`;
		
		angleIndicator.style.display = "block";
		angleIndicator.style.left = `${rect.width / 2 + normalizedX * maxDistance}px`;
		angleIndicator.style.top = `${rect.height / 2 - normalizedY * maxDistance}px`;
		
		runForwardFunction();
	});
}