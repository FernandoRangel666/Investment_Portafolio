// test-api.js

// Modern Node.js has global fetch, so no need to install anything if using Node >= 18
// If you're using an older version, run: npm install node-fetch

const API_BASE = 'http://localhost:5000';

async function testPredictionGET(ticker) {
    console.log(`\n🔍 Testing GET request for ticker: ${ticker}`);
    try {
        const response = await fetch(`${API_BASE}/predict?ticker=${ticker}`);
        const data = await response.json();
        console.log('✅ Success! Response:', JSON.stringify(data, null, 2));
    } catch (error) {
        console.error('❌ Error calling API:', error.message);
    }
}

async function testPredictionPOST(ticker, dataPath, modelPath) {
    console.log(`\n📤 Testing POST request for ticker: ${ticker}`);
    try {
        const payload = { ticker };
        if (dataPath) payload.data_path = dataPath;
        if (modelPath) payload.model_path = modelPath;

        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        console.log('✅ Success! Response:', JSON.stringify(data, null, 2));
    } catch (error) {
        console.error('❌ Error calling API:', error.message);
    }
}

async function checkHealth() {
    console.log('\n🏥 Checking API health...');
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        console.log('✅ Health check:', data);
    } catch (error) {
        console.error('❌ Health check failed:', error.message);
    }
}

// Main test execution
async function main() {
    console.log('🧪 Starting API tests...');

    // First, check if the server is up
    await checkHealth();

    // Test GET
    await testPredictionGET('AAPL');

    // Test POST with custom paths (optional)
    await testPredictionPOST('NVDA', 'data/stock_data.pkl', '../models/ppo_portfolio_trading');

    // Try invalid ticker
    await testPredictionGET('INVALID');
}

// Run the tests
main();