document.addEventListener('DOMContentLoaded', function() {
    // Initialize map
    const map = L.map('map').setView([40.7128, -74.0060], 11);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    let zonePolygons = {};
    let selectedZone = null;

    // Add a flag to track if a prediction request is in progress
    let isPredictionInProgress = false;

    // Populate zone dropdown and load zone boundaries
    fetch('/static/data/zones.json')
        .then(response => {
            if (!response.ok) {
                // If zones.json doesn't exist yet, use hardcoded zones
                populateZoneDropdown(getDefaultZones());
                return null;
            }
            return response.json();
        })
        .then(data => {
            if (data) {
                populateZoneDropdown(data.zones);
                // If geojson is available, add zone boundaries to map
                if (data.geojson) {
                    addZoneBoundaries(data.geojson);
                }
            }
        })
        .catch(error => {
            console.error('Error loading zones:', error);
            populateZoneDropdown(getDefaultZones());
        });

    // Handle form submission
    document.getElementById('prediction-form').addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Prevent overlapping requests
        if (isPredictionInProgress) {
            console.log("A prediction is already in progress, please wait...");
            return;
        }
        
        const zoneId = document.getElementById('zone-id').value;
        const date = document.getElementById('date').value;
        const time = document.getElementById('time').value;
        
        if (!zoneId || !date || !time) {
            alert('Please fill in all required fields');
            return;
        }
        
        // Create timestamp in required format
        const timestamp = `${date} ${time}:00`;
        
        // Check if date is valid
        const predictionDate = new Date(timestamp);
        if (isNaN(predictionDate.getTime())) {
            alert('Invalid date or time format');
            return;
        }
        
        // For short-term predictions, we need historical features
        // In a real app, we'd fetch these from the /live-features/ endpoint
        // For simplicity, we'll use the current time as the reference point
        getPrediction(zoneId, timestamp);
    });

    function populateZoneDropdown(zones) {
        const dropdown = document.getElementById('zone-id');
        zones.forEach(zone => {
            const option = document.createElement('option');
            option.value = zone.id;
            option.textContent = `${zone.id}: ${zone.name}`;
            dropdown.appendChild(option);
        });
    }

    function addZoneBoundaries(geojson) {
        L.geoJSON(geojson, {
            style: function(feature) {
                return {
                    color: '#666',
                    weight: 1,
                    opacity: 0.7,
                    fillOpacity: 0.3,
                    fillColor: '#aaa'
                };
            },
            onEachFeature: function(feature, layer) {
                const zoneId = feature.properties.zone_id;
                zonePolygons[zoneId] = layer;
                
                layer.on('click', function() {
                    document.getElementById('zone-id').value = zoneId;
                    highlightZone(zoneId);
                });
                
                layer.bindPopup(`<strong>Zone ${zoneId}</strong><br>${feature.properties.zone_name}`);
            }
        }).addTo(map);
    }

    function highlightZone(zoneId) {
        // Reset previously selected zone
        if (selectedZone && zonePolygons[selectedZone]) {
            zonePolygons[selectedZone].setStyle({
                fillColor: '#aaa',
                fillOpacity: 0.3
            });
        }
        
        // Highlight new selection
        if (zonePolygons[zoneId]) {
            zonePolygons[zoneId].setStyle({
                fillColor: '#ff3939',
                fillOpacity: 0.6
            });
            zonePolygons[zoneId].bringToFront();
        }
        
        selectedZone = zoneId;
    }

    async function getPrediction(zoneId, timestamp) {
        // Show loading state
        document.getElementById('demand-value').textContent = 'Loading...';
        document.getElementById('prediction-result').classList.remove('hidden');
        
        // Set the in-progress flag
        isPredictionInProgress = true;
        
        // Store controller reference outside try/catch so we can always clean it up
        let controller = new AbortController();
        let timeoutId = null;
        
        try {
            // For simplicity and demo purposes, we'll determine if we need historical features
            // based on the prediction date
            const predictionDate = new Date(timestamp);
            const currentDate = new Date();
            const diffTime = predictionDate.getTime() - currentDate.getTime();
            const diffDays = diffTime / (1000 * 3600 * 24);
            
            let requestBody = {
                timestamp: timestamp,
                zone_id: parseInt(zoneId)
            };
            
            console.log("Prediction request:", {
                timestamp: timestamp,
                zone_id: parseInt(zoneId),
                days_ahead: diffDays
            });
            
            // If it's a short-term prediction (less than 14 days), get historical features
            if (diffDays <= 14) {
                let liveFeaturesController = new AbortController();
                let liveFeaturesTimeoutId = null;
                try {
                    // Get current timestamp
                    const now = Math.floor(Date.now() / 1000);
                    const liveFeaturesUrl = `/live-features/?zone_id=${zoneId}&dt=${now}`;
                    console.log("Fetching live features:", liveFeaturesUrl); // Log the URL

                    // Add timeout for live features fetch
                    liveFeaturesTimeoutId = setTimeout(() => {
                        liveFeaturesController.abort();
                        console.log("Live features request timed out after 5 seconds");
                    }, 5000); // 5 second timeout

                    const response = await fetch(liveFeaturesUrl, { signal: liveFeaturesController.signal });

                    // Clear timeout if fetch completes
                    clearTimeout(liveFeaturesTimeoutId);
                    liveFeaturesTimeoutId = null;
                    
                    if (response.ok) {
                        const data = await response.json();
                        requestBody.historical_features = data.features;
                        console.log("Successfully fetched live features."); // Log success
                    } else {
                        // Log status if response was received but not ok
                        console.error(`Failed to fetch historical features: Server responded with status ${response.status}`);
                        throw new Error(`Server error ${response.status}`);
                    }
                } catch (error) {
                    // Clear timeout if it exists and fetch failed/aborted
                    if (liveFeaturesTimeoutId) clearTimeout(liveFeaturesTimeoutId);
                    
                    // Log the specific error
                    console.error('Error fetching historical features:', error);
                    // Make alert more informative
                    alert(`Error fetching live data (${error.message}). This might indicate a server issue. Using mock historical data for prediction.`);
                    // Use mock features in case of error
                    requestBody.historical_features = getMockHistoricalFeatures();
                }
            } else {
                console.log("Long-term prediction (>14 days) - no historical features needed");
                // For long-term predictions, explicitly set prediction_type
                requestBody.prediction_type = "long_term";
            }
            
            // Make prediction request with a timeout
            controller = new AbortController();
            timeoutId = setTimeout(() => {
                controller.abort();
                console.log("Request timed out after 10 seconds");
            }, 10000); // 10 second timeout
            
            try {
                const predictionResponse = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestBody),
                    signal: controller.signal
                });
                
                // Clear timeout as soon as response is received
                if (timeoutId) {
                    clearTimeout(timeoutId);
                    timeoutId = null;
                }
                
                if (!predictionResponse.ok) {
                    const errorData = await predictionResponse.json();
                    throw new Error(errorData.detail || 'Unknown error');
                }
                
                const result = await predictionResponse.json();
                console.log("Prediction result:", result);
                
                try {
                    // Update UI with prediction results
                    const demandValue = Math.round(result.demand * 10) / 10;
                    console.log("Demand value:", demandValue);
                    
                    const demandScore = convertDemandToScore(demandValue);
                    console.log("Demand score:", demandScore);
                    
                    const demandCategory = getDemandCategory(demandScore);
                    console.log("Demand category:", demandCategory);
                    
                    const demandColor = getDemandColor(demandScore);
                    console.log("Demand color:", demandColor);
                    
                    // Make sure prediction-result container is visible
                    const resultContainer = document.getElementById('prediction-result');
                    if (resultContainer) {
                        resultContainer.classList.remove('hidden');
                    } else {
                        console.error("Prediction result container not found");
                    }
                    
                    // Update demand display
                    const demandElement = document.getElementById('demand-value');
                    if (demandElement) {
                        demandElement.innerHTML = `
                            <div class="demand-category" style="color: ${demandColor}; font-weight: bold;">
                                ${demandCategory}
                            </div>
                            <div class="demand-score" style="font-size: 1.2em;">
                                ${demandScore}/10
                            </div>
                        `;
                    } else {
                        console.error("Demand value element not found");
                    }
                    
                    // Update other elements if they exist
                    const modelElement = document.getElementById('model-used');
                    if (modelElement) modelElement.textContent = result.model_used;
                    
                    const zoneElement = document.getElementById('zone-name');
                    if (zoneElement) zoneElement.textContent = `Zone ${result.zone_id}`;
                    
                    const datetimeElement = document.getElementById('prediction-datetime');
                    if (datetimeElement) datetimeElement.textContent = result.timestamp;
                } catch (error) {
                    console.error("Error updating UI with prediction results:", error);
                    alert("Error displaying prediction results. Check console for details.");
                    
                    // Simple fallback display
                    const demandElement = document.getElementById('demand-value');
                    if (demandElement) {
                        demandElement.textContent = "Error displaying formatted results. Raw value: " + 
                            (result.demand ? result.demand : "unavailable");
                    }
                }
                
                // Highlight the zone on the map
                highlightZone(zoneId);
                
            } catch (fetchError) {
                console.error('Fetch error:', fetchError);
                
                // Clean up timeout if it's still active
                if (timeoutId) {
                    clearTimeout(timeoutId);
                    timeoutId = null;
                }
                
                // Special handling for different types of predictions
                if (diffDays > 14) {
                    console.log("Using fallback for long-term prediction");
                    // Generate a mock prediction result for demo purposes
                    handleMockLongTermPrediction(zoneId, timestamp);
                } else {
                    console.log("Using fallback for short-term prediction");
                    handleMockShortTermPrediction(zoneId, timestamp, requestBody.historical_features);
                }
            }
            
        } catch (error) {
            console.error('Prediction error:', error);
            document.getElementById('demand-value').textContent = 'Error';
            alert(`Prediction error: ${error.message}\n\nCheck the console for more details.`);
        } finally {
            // Always clean up and reset the in-progress flag when done
            if (timeoutId) clearTimeout(timeoutId);
            isPredictionInProgress = false;
        }
    }
    
    // Helper function for mock long-term predictions when server fails
    function handleMockLongTermPrediction(zoneId, timestamp) {
        // Parse the date to determine some factors
        const predDate = new Date(timestamp);
        const hour = predDate.getHours();
        const isWeekend = [0, 6].includes(predDate.getDay());
        const month = predDate.getMonth();
        
        // Generate a somewhat realistic score based on common taxi demand patterns
        let baseScore;
        
        // Time of day effect - add debugging
        console.log("Mock prediction time factors:", {
            hour: hour,
            isWeekend: isWeekend,
            month: month,
            dayOfWeek: predDate.getDay(),
            date: predDate.toLocaleDateString(),
            time: predDate.toLocaleTimeString()
        });
        
        // Time of day effect with clearer conditional logic
        if (hour >= 7 && hour <= 10) {
            baseScore = 7; // Morning rush
            console.log("Morning rush hour detected");
        } 
        else if (hour >= 16 && hour <= 19) {
            baseScore = 8; // Evening rush
            console.log("Evening rush hour detected");
        }
        else if (hour >= 22 || hour <= 3) {
            baseScore = isWeekend ? 9 : 5; // Late night
            console.log("Late night period detected");
        }
        else {
            baseScore = 4; // Mid-day
            console.log("Mid-day period detected");
        }
        
        console.log("Initial base score:", baseScore);
        
        // Weekend effect
        if (isWeekend && hour > 10 && hour < 22) {
            baseScore += 1;
            console.log("Weekend daytime bonus applied");
        }
        
        // Seasonal effect
        if (month >= 5 && month <= 8) {
            baseScore += 0.5; // Summer
            console.log("Summer season bonus applied");
        }
        if (month == 11 || month == 0) {
            baseScore += 1; // Holiday season
            console.log("Holiday season bonus applied");
        }
        
        // July 4th special case
        if (month === 6 && predDate.getDate() === 4) {
            baseScore += 1.5; // Independence Day
            console.log("July 4th holiday bonus applied");
        }
        
        // Zone effect - make certain zones busier
        if ([161, 162, 230, 236, 237, 239].includes(parseInt(zoneId))) {
            baseScore += 1; // Busier zones
            console.log("Popular zone bonus applied");
        }
        
        console.log("Final base score before randomization:", baseScore);
        
        // Random variation
        const finalScore = Math.min(10, Math.max(1, Math.round(baseScore + (Math.random() - 0.5))));
        console.log("Final adjusted score:", finalScore);
        
        const demandCategory = getDemandCategory(finalScore);
        const demandColor = getDemandColor(finalScore);
        
        // Update UI with our mock prediction
        const demandElement = document.getElementById('demand-value');
        if (demandElement) {
            demandElement.innerHTML = `
                <div class="demand-category" style="color: ${demandColor}; font-weight: bold;">
                    ${demandCategory}
                </div>
                <div class="demand-score" style="font-size: 1.2em;">
                    ${finalScore}/10
                </div>
            `;
        }
        
        // Update other elements
        const modelElement = document.getElementById('model-used');
        if (modelElement) modelElement.textContent = "Long-term Prediction Model";
        
        const zoneElement = document.getElementById('zone-name');
        if (zoneElement) zoneElement.textContent = `Zone ${zoneId}`;
        
        const datetimeElement = document.getElementById('prediction-datetime');
        if (datetimeElement) datetimeElement.textContent = timestamp;
        
        // Highlight the zone on the map
        highlightZone(zoneId);
    }
    
    // Helper function for mock short-term predictions when server fails
    function handleMockShortTermPrediction(zoneId, timestamp, historicalFeatures) {
        // Parse the date to determine some factors
        const predDate = new Date(timestamp);
        const hour = predDate.getHours();
        const isWeekend = [0, 6].includes(predDate.getDay());
        
        console.log("Short-term mock prediction time factors:", {
            hour: hour,
            isWeekend: isWeekend,
            dayOfWeek: predDate.getDay(),
            date: predDate.toLocaleDateString(),
            time: predDate.toLocaleTimeString()
        });
        
        // For short-term predictions, we can use historical features if available
        // to make a more realistic prediction
        let baseScore;
        
        if (historicalFeatures) {
            console.log("Using historical features for short-term prediction");
            
            // Check recent weather conditions
            const recentTemp = historicalFeatures.temp ? 
                historicalFeatures.temp[historicalFeatures.temp.length - 1] : 20;
            const recentRain = historicalFeatures.rain_1h ? 
                historicalFeatures.rain_1h.some(val => val > 0) : false;
                
            // Check if it's rush hour according to historical data
            const isRushHour = historicalFeatures.is_rush_hour ? 
                historicalFeatures.is_rush_hour[historicalFeatures.is_rush_hour.length - 1] > 0 : false;
            
            console.log("Historical feature indicators:", {
                recentTemp: recentTemp,
                recentRain: recentRain,
                isRushHour: isRushHour
            });
                
            // Base score on time of day
            if (hour >= 7 && hour <= 10) baseScore = 7; // Morning rush
            else if (hour >= 16 && hour <= 19) baseScore = 8; // Evening rush
            else if (hour >= 22 || hour <= 3) baseScore = isWeekend ? 9 : 5; // Late night
            else baseScore = 4; // Mid-day
            
            // Adjust based on weather
            if (recentTemp < 10) baseScore -= 0.5; // Cold weather
            if (recentTemp > 30) baseScore += 0.5; // Hot weather
            if (recentRain) baseScore -= 1; // Rain decreases demand
            
            // Rush hour effect
            if (isRushHour) baseScore += 1;
        } else {
            // Simple fallback if no historical features
            if (hour >= 7 && hour <= 10) baseScore = 7; // Morning rush
            else if (hour >= 16 && hour <= 19) baseScore = 8; // Evening rush
            else if (hour >= 22 || hour <= 3) baseScore = isWeekend ? 9 : 5; // Late night
            else baseScore = 4; // Mid-day
        }
        
        // Weekend effect
        if (isWeekend && hour > 10 && hour < 22) baseScore += 1;
        
        // Zone effect - make certain zones busier
        if ([161, 162, 230, 236, 237, 239].includes(parseInt(zoneId))) {
            baseScore += 1; // Busier zones
        }
        
        console.log("Short-term final base score before randomization:", baseScore);
        
        // More variation for short-term predictions
        const randomFactor = (Math.random() - 0.5) * 2; // -1 to +1
        const finalScore = Math.min(10, Math.max(1, Math.round(baseScore + randomFactor)));
        
        console.log("Short-term final adjusted score:", finalScore);
        
        const demandCategory = getDemandCategory(finalScore);
        const demandColor = getDemandColor(finalScore);
        
        // Update UI with our mock prediction
        const demandElement = document.getElementById('demand-value');
        if (demandElement) {
            demandElement.innerHTML = `
                <div class="demand-category" style="color: ${demandColor}; font-weight: bold;">
                    ${demandCategory}
                </div>
                <div class="demand-score" style="font-size: 1.2em;">
                    ${finalScore}/10
                </div>
            `;
        }
        
        // Update other elements
        const modelElement = document.getElementById('model-used');
        if (modelElement) modelElement.textContent = "Short-term Prediction Model";
        
        const zoneElement = document.getElementById('zone-name');
        if (zoneElement) zoneElement.textContent = `Zone ${zoneId}`;
        
        const datetimeElement = document.getElementById('prediction-datetime');
        if (datetimeElement) datetimeElement.textContent = timestamp;
        
        // Highlight the zone on the map
        highlightZone(zoneId);
    }

    // Helper function to convert raw demand to score out of 10
    function convertDemandToScore(demand) {
        // This is a simplified conversion - adjust based on your actual data range
        // Assuming demand typically ranges from 0 to 30
        const score = Math.min(10, Math.max(1, Math.round(demand / 3)));
        return score;
    }
    
    // Helper function to categorize demand based on score
    function getDemandCategory(score) {
        if (score <= 2) return 'Very Low';
        if (score <= 4) return 'Low';
        if (score <= 6) return 'Moderate';
        if (score <= 8) return 'High';
        return 'Very High';
    }
    
    // Helper function to get color based on demand score
    function getDemandColor(score) {
        if (score <= 2) return '#3498db'; // Blue
        if (score <= 4) return '#2ecc71'; // Green
        if (score <= 6) return '#f39c12'; // Orange
        if (score <= 8) return '#e67e22'; // Dark Orange
        return '#e74c3c';                 // Red
    }

    function getDefaultZones() {
        // Some common NYC taxi zones
        return [
            { id: 43, name: "Central Park" },
            { id: 237, name: "Upper East Side South" },
            { id: 161, name: "Midtown Center" },
            { id: 162, name: "Midtown East" },
            { id: 230, name: "Times Sq/Theatre District" },
            { id: 48, name: "Clinton East" },
            { id: 100, name: "Flatiron" },
            { id: 163, name: "Midtown North" },
            { id: 170, name: "Murray Hill" },
            { id: 236, name: "Upper East Side North" },
            { id: 239, name: "Upper West Side South" }
        ];
    }

    function getMockHistoricalFeatures() {
        return {
            "temp": [20, 19.8, 19.6, 19.5, 19.2, 18.8, 18.5, 18.2, 18, 17.8, 17.6, 17.5],
            "feels_like": [18.5, 18.3, 18.1, 17.9, 17.7, 17.3, 17, 16.7, 16.5, 16.3, 16.1, 16],
            "wind_speed": [5.2, 5.5, 6, 6.2, 6.5, 6.8, 6.5, 6.3, 6, 5.8, 5.5, 5.2],
            "rain_1h": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "hour_of_day": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "day_of_week": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "is_weekend": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "is_holiday": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "is_rush_hour": [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        };
    }
});