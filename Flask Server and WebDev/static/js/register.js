document.addEventListener('DOMContentLoaded', function () {
    const registerForm = document.getElementById('registerForm');
    const countryCodeSelect = document.getElementById('countryCode');
    const phoneNumberInput = document.getElementById('phoneNumber');

    const errorMessage = document.createElement('div');
    errorMessage.className = 'error-message';
    registerForm.appendChild(errorMessage);

    registerForm.addEventListener('submit', function (e) {
        e.preventDefault();

        // Get form values
        const name = document.getElementById('fullName').value;
        const phoneNumber = document.getElementById('phoneNumber').value;
        const numberPlate = document.getElementById('numberPlate').value;
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        // Role is now hardcoded to 'client' on this page
        const role = 'client'; 

        // Basic validation
        if (password !== confirmPassword) {
            errorMessage.textContent = 'Passwords do not match';
            errorMessage.style.display = 'block';
            return;
        }

        // Show loading state
        const submitButton = registerForm.querySelector('button[type="submit"]');
        submitButton.disabled = true;
        submitButton.textContent = 'Registering...';

        // Update the formData creation
        const formData = {
            name: name,
            phoneNumber: countryCodeSelect.value + phoneNumber, // Include country code
            numberPlate: numberPlate,
            password: password,
            role: role
        };

        // Send registration request
        fetch('/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(formData)
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    window.location.href = '/login';
                } else {
                    errorMessage.textContent = data.message || 'Registration failed';
                    errorMessage.style.display = 'block';
                }
            })
            .catch(error => {
                console.error('Registration error:', error);
                errorMessage.textContent = 'An error occurred during registration';
                errorMessage.style.display = 'block';
            })
            .finally(() => {
                submitButton.disabled = false;
                submitButton.textContent = 'Register';
            });
    });
});