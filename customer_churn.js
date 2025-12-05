// --- DUMMY DATA FOR CHART.JS (Radar) ---
// Note: Initial data set to zeros; the real data is populated from the API response.
const radarChartConfig = {
    labels: ['Tenure Score', 'Charges Score', 'Service Usage', 'Contract Type', 'Support Index', 'Streaming'],
    datasets: [{
        label: 'Risk Score (Higher is Worse)',
        data: [0, 0, 0, 0, 0, 0], // Placeholder data, updated by API response
        backgroundColor: 'rgba(255, 152, 0, 0.3)', 
        borderColor: '#ff9800',
        pointBackgroundColor: '#ff9800',
        pointBorderColor: '#fff',
    }]
};

const radarOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
        r: {
            angleLines: { color: 'rgba(255, 255, 255, 0.2)' },
            grid: { color: 'rgba(255, 255, 255, 0.2)' },
            pointLabels: { color: '#f0f0f0', font: { size: 12, weight: 'bold' } },
            suggestedMin: 0,
            suggestedMax: 1,
            ticks: { display: false, stepSize: 0.2 }
        }
    },
    plugins: {
        legend: { display: false }
    }
};

let radarChartInstance = null;

// The previous DUMMY ACTION CARDS DATA has been removed.

// --- MAIN PREDICTION HANDLER ---
document.getElementById('churnForm').addEventListener('submit', async function(event) {
    event.preventDefault(); 
    
    const form = event.target;
    const inputSection = document.getElementById('inputSection');
    const dashboardSection = document.getElementById('dashboardSection');
    const loadingSpinner = document.getElementById('loadingSpinner');
    
    const formData = new FormData(form);
    const rawData = Object.fromEntries(formData.entries());

    // CRITICAL: Prepare numerical data types for the model payload
    const inputData = {
        ...rawData,
        // Send all new and existing fields
        'customerID': rawData.customerID, 
        'PhoneService': rawData.PhoneService,
        'MultipleLines': rawData.MultipleLines,
        'DeviceProtection': rawData.DeviceProtection,
        'StreamingTV': rawData.StreamingTV,
        
        // Convert numerical fields
        'tenure': parseInt(rawData.tenure),
        'MonthlyCharges': parseFloat(rawData.MonthlyCharges),
        // TotalCharges needs special handling for empty string if form wasn't interacted with, but required=true helps
        'TotalCharges': parseFloat(rawData.TotalCharges),
        'SeniorCitizen': parseInt(rawData.SeniorCitizen)
    };

    // UI State: Hide Input, Show Loading
    inputSection.classList.add('hidden');
    loadingSpinner.classList.remove('hidden');
    dashboardSection.classList.add('hidden'); 

    // --- API ENDPOINT (CONNECT FLASK HERE) ---
    const API_ENDPOINT = 'http://127.0.0.1:5000/predict'; 

    try {
        // --- REAL API CALL (Replaced Simulation Block) ---
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(inputData)
        });

        if (!response.ok) {
            // Throw an error if the HTTP status is not 2xx
            const errorText = await response.text();
            throw new Error(`HTTP error! Status: ${response.status}. Response: ${errorText}`);
        }

        const apiResponse = await response.json();
        // -------------------------------------------------------------------


        // --- UI State: Process and Display ---
        loadingSpinner.classList.add('hidden');
        dashboardSection.classList.remove('hidden');

        // 1. TOP KPI CARDS
        const probability = apiResponse.probability;
        // const isChurn = apiResponse.prediction === 1; // Not used but good to have
        const riskLevel = probability < 0.35 ? 'LOW' : (probability < 0.65 ? 'MEDIUM' : 'HIGH');
        
        document.getElementById('kpi-probability').textContent = `${(probability * 100).toFixed(1)}%`;
        
        const riskBadge = document.getElementById('kpi-risk-badge');
        riskBadge.textContent = riskLevel;
        riskBadge.className = `kpi-value risk-${riskLevel.toLowerCase()}`;

        // Confidence score calculation is a proxy: 100% - |probability - 0.5| * 200
        document.getElementById('kpi-confidence').textContent = `${(100 - Math.abs(probability - 0.5) * 200).toFixed(0)}%`;
        // Revenue at risk: Monthly Charges * (e.g., 12 months) * Churn Probability
        document.getElementById('kpi-revenue').textContent = `$${(inputData.MonthlyCharges * 12 * probability).toFixed(0)}`;


        // 2. FEATURE CONTRIBUTION TABLE
        const tableBody = document.querySelector('#contribution-table tbody');
        tableBody.innerHTML = ''; 
        // Ensure your API returns 'feature_contributions' in the expected format
        if (apiResponse.feature_contributions && Array.isArray(apiResponse.feature_contributions)) {
            apiResponse.feature_contributions.forEach(item => {
                const scoreClass = item.score > 0.1 ? 'score-high-risk' : (item.score < -0.1 ? 'score-low-risk' : 'score-neutral');
                const row = tableBody.insertRow();
                
                row.insertCell().textContent = item.feature;
                
                const scoreCell = row.insertCell();
                scoreCell.className = scoreClass;
                scoreCell.textContent = item.score.toFixed(3);
                
                row.insertCell().textContent = item.explanation;
            });
        }

        // 3. FUNNEL VISUALIZATION
        document.querySelectorAll('.funnel-stage').forEach(el => el.classList.remove('highlighted'));

        let targetStageId;
        if (probability < 0.3) targetStageId = 'stage-active';
        else if (probability < 0.6) targetStageId = 'stage-at-risk';
        else if (probability < 0.9) targetStageId = 'stage-likely-to-churn';
        else targetStageId = 'stage-churned';
        
        document.getElementById(targetStageId).classList.add('highlighted');
        
        // 4. RADAR CHART (Chart.js Initialization)
        if (radarChartInstance) {
            radarChartInstance.destroy();
        }
        
        // Ensure your API returns 'radar_scores' in the expected format (e.g., {tenure: 0.5, charges: 0.4, ...})
        if (apiResponse.radar_scores) {
            const radarScores = apiResponse.radar_scores;
            radarChartConfig.datasets[0].data = [
                radarScores.tenure,
                radarScores.charges,
                radarScores.total_charges,
                radarScores.service_usage,
                radarScores.contract_type,
                radarScores.support_availability
            ];

            const ctx = document.getElementById('radarChart').getContext('2d');
            radarChartInstance = new Chart(ctx, {
                type: 'radar',
                data: radarChartConfig,
                options: radarOptions
            });
        }

        // 5. AI Recommended Actions: Logic removed as requested and HTML section is commented out.


    } catch (error) {
        // Handle API connection or parsing errors
        loadingSpinner.classList.add('hidden');
        inputSection.classList.remove('hidden');
        document.querySelector('.submit-btn').innerHTML = 'PREDICT CHURN';
        
        // Display an error message to the user
        const existingError = document.querySelector('.api-error-note');
        if(existingError) existingError.remove();

        const errorNote = document.createElement('div');
        errorNote.className = 'result-box high-risk card api-error-note';
        errorNote.innerHTML = `
            <h2>API Connection Error 🛑</h2>
            <p>Could not process prediction. Please ensure your **Flask Backend** is running and returns the expected JSON format.</p>
            <p style="font-size: 0.8em;">(Error: ${error.message})</p>
        `;
        document.querySelector('.container').appendChild(errorNote);
    }
});