// Simple test to verify frontend-backend connection
console.log('Testing backend connection...');

// Test 1: Health check
fetch('http://localhost:8000/api/v1/health')
  .then(response => response.json())
  .then(data => {
    console.log('✅ Backend health check successful:', data);
  })
  .catch(error => {
    console.error('❌ Backend health check failed:', error);
  });

// Test 2: Check if CORS is working
fetch('http://localhost:8000/api/v1/health', {
  method: 'GET',
  headers: {
    'Content-Type': 'application/json'
  }
})
.then(response => {
  console.log('✅ CORS check successful, status:', response.status);
  return response.json();
})
.then(data => {
  console.log('✅ API response:', data);
})
.catch(error => {
  console.error('❌ CORS/API connection failed:', error);
});

console.log('Connection tests initiated. Check console for results.'); 