import http from 'k6/http';
import { sleep, check } from 'k6';

// Test configuration
export const options = {
  // Stages for ramping up and down load
  stages: [
    { duration: '30s', target: 10 }, // Ramp up to 10 users over 30 seconds
    { duration: '1m', target: 10 },  // Stay at 10 users for 1 minute
    { duration: '30s', target: 20 }, // Ramp up to 20 users over 30 seconds
    { duration: '1m', target: 20 },  // Stay at 20 users for 1 minute
    { duration: '30s', target: 0 },  // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests should complete below 2s
    http_req_failed: ['rate<0.01'],    // Less than 1% of requests should fail
  },
};

// API endpoint and authentication
const API_URL = 'https://api.mykb.online/v1/embeddings';
const API_TOKEN = 'YOUR_BEARER_TOKEN_HERE'; // Replace with your actual token

// Test data
const SINGLE_TEXT_PAYLOAD = JSON.stringify({
  input: 'This is a test sentence for benchmarking the API under load.',
  model: 'sentence-transformers/all-MiniLM-L6-v2'
});

const BATCH_TEXT_PAYLOAD = JSON.stringify({
  input: [
    'First sentence for batch testing under load.',
    'Second sentence for batch testing under load.',
    'Third sentence for batch testing under load.',
    'Fourth sentence for batch testing under load.',
    'Fifth sentence for batch testing under load.'
  ],
  model: 'BAAI/bge-m3'
});

// Default request parameters
const params = {
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_TOKEN}`
  },
};

export default function () {
  // Randomly choose between single text and batch text to simulate varied load
  const payload = Math.random() < 0.7 ? SINGLE_TEXT_PAYLOAD : BATCH_TEXT_PAYLOAD;
  
  // Send the request
  const response = http.post(API_URL, payload, params);
  
  // Check if the request was successful
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response has data': (r) => r.json('data') !== undefined,
    'response has correct model': (r) => {
      const body = r.json();
      const requestBody = JSON.parse(payload);
      return body.model === requestBody.model;
    },
  });
  
  // Add some sleep time between requests to simulate real-world usage
  sleep(1);
}

// Helper function to log test progress
export function setup() {
  console.log('Starting load test for Memementor Embedding Service');
  return {};
}

export function teardown(data) {
  console.log('Load test completed');
}
