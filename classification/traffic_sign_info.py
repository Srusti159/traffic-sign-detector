"""
Traffic Sign Knowledge Base for GTSRB 43 Classes.
Provides human-readable interpretation, safety meaning, and recommended driver actions.
"""

GTSRB_CLASSES = {
    0: {
        "name": "Speed limit (20km/h)",
        "meaning": "Maximum permitted speed is 20 km/h.",
        "action": "Drive at or below 20 km/h; watch for pedestrians."
    },
    1: {
        "name": "Speed limit (30km/h)",
        "meaning": "Maximum permitted speed is 30 km/h.",
        "action": "Maintain speed within 30 km/h."
    },
    2: {
        "name": "Speed limit (50km/h)",
        "meaning": "Maximum permitted speed is 50 km/h (Standard urban speed limit).",
        "action": "Maintain speed within 50 km/h."
    },
    3: {
        "name": "Speed limit (60km/h)",
        "meaning": "Maximum permitted speed is 60 km/h.",
        "action": "Maintain speed within 60 km/h."
    },
    4: {
        "name": "Speed limit (70km/h)",
        "meaning": "Maximum permitted speed is 70 km/h.",
        "action": "Maintain speed within 70 km/h."
    },
    5: {
        "name": "Speed limit (80km/h)",
        "meaning": "Maximum permitted speed is 80 km/h.",
        "action": "Maintain speed within 80 km/h."
    },
    6: {
        "name": "End of speed limit (80km/h)",
        "meaning": "End of 80 km/h speed restriction zone.",
        "action": "Standard rural speed limit applies unless otherwise posted."
    },
    7: {
        "name": "Speed limit (100km/h)",
        "meaning": "Maximum permitted speed is 100 km/h.",
        "action": "Maintain speed within 100 km/h."
    },
    8: {
        "name": "Speed limit (120km/h)",
        "meaning": "Maximum permitted speed is 120 km/h (Motorway limit).",
        "action": "Maintain speed within 120 km/h."
    },
    9: {
        "name": "No passing",
        "meaning": "Overtaking other vehicles is prohibited.",
        "action": "Do not overtake any multi-track motor vehicles."
    },
    10: {
        "name": "No passing for vehicles over 3.5 metric tons",
        "meaning": "Heavy vehicles > 3.5t prohibited from passing.",
        "action": "Trucks must stay in lane; passenger cars may pass."
    },
    11: {
        "name": "Right-of-way at the next intersection",
        "meaning": "You have priority right-of-way at the next upcoming intersection.",
        "action": "Proceed with caution; cross-traffic must yield."
    },
    12: {
        "name": "Priority road",
        "meaning": "Priority road (you have continuous right-of-way).",
        "action": "Continue driving; vehicles on intersecting side roads must yield."
    },
    13: {
        "name": "Yield",
        "meaning": "Yield right-of-way to cross-traffic.",
        "action": "Slow down and stop if necessary to allow other traffic to pass."
    },
    14: {
        "name": "Stop",
        "meaning": "Full stop mandatory before proceeding.",
        "action": "Come to a complete stop at the stop line; yield to all traffic."
    },
    15: {
        "name": "No vehicles",
        "meaning": "Road closed to all vehicles in both directions.",
        "action": "Do not enter the roadway."
    },
    16: {
        "name": "Vehicles over 3.5 metric tons prohibited",
        "meaning": "Vehicles exceeding 3.5 metric tons prohibited.",
        "action": "Commercial heavy trucks must find an alternate route."
    },
    17: {
        "name": "No entry",
        "meaning": "No entry (One-way street in opposite direction).",
        "action": "Do NOT enter; turn around or take another road."
    },
    18: {
        "name": "General caution",
        "meaning": "General caution / Danger ahead.",
        "action": "Reduce speed, pay increased attention to road conditions."
    },
    19: {
        "name": "Dangerous curve to the left",
        "meaning": "Sharp curve to the left ahead.",
        "action": "Slow down before entering the leftward curve."
    },
    20: {
        "name": "Dangerous curve to the right",
        "meaning": "Sharp curve to the right ahead.",
        "action": "Slow down before entering the rightward curve."
    },
    21: {
        "name": "Double curve",
        "meaning": "Double curve ahead (first to left or right).",
        "action": "Reduce speed for successive sharp curves."
    },
    22: {
        "name": "Bumpy road",
        "meaning": "Bumpy or uneven road surface ahead.",
        "action": "Reduce speed to prevent vehicle damage or loss of control."
    },
    23: {
        "name": "Slippery road",
        "meaning": "Slippery road condition ahead (rain, ice, mud).",
        "action": "Avoid abrupt steering or harsh braking."
    },
    24: {
        "name": "Road narrows on the right",
        "meaning": "Road narrows on the right side.",
        "action": "Merge smoothly and watch for oncoming traffic."
    },
    25: {
        "name": "Road work",
        "meaning": "Road construction / maintenance work in progress.",
        "action": "Slow down, watch for workers and construction machinery."
    },
    26: {
        "name": "Traffic signals",
        "meaning": "Traffic light signals ahead.",
        "action": "Prepare to stop if the upcoming signal is yellow or red."
    },
    27: {
        "name": "Pedestrians",
        "meaning": "Pedestrian crossing area ahead.",
        "action": "Slow down and be ready to yield to crossing pedestrians."
    },
    28: {
        "name": "Children crossing",
        "meaning": "Children crossing / School zone nearby.",
        "action": "Drive slowly and be vigilant for children near the road."
    },
    29: {
        "name": "Bicycles crossing",
        "meaning": "Bicycle crossing ahead.",
        "action": "Watch for cyclists entering or crossing the road."
    },
    30: {
        "name": "Beware of ice/snow",
        "meaning": "Beware of snow or ice on road.",
        "action": "Exercise extreme caution; winter tires or chains may be required."
    },
    31: {
        "name": "Wild animals crossing",
        "meaning": "Wild animal crossing zone.",
        "action": "Be alert for animals crossing, especially during dawn and dusk."
    },
    32: {
        "name": "End of all speed and passing limits",
        "meaning": "End of all previously posted speed and passing restrictions.",
        "action": "Standard highway rules apply."
    },
    33: {
        "name": "Turn right ahead",
        "meaning": "Mandatory turn right ahead.",
        "action": "Turn right at the upcoming intersection."
    },
    34: {
        "name": "Turn left ahead",
        "meaning": "Mandatory turn left ahead.",
        "action": "Turn left at the upcoming intersection."
    },
    35: {
        "name": "Ahead only",
        "meaning": "Mandatory straight-ahead direction only.",
        "action": "Do not turn left or right; proceed straight."
    },
    36: {
        "name": "Go straight or right",
        "meaning": "Allowed to travel straight or turn right.",
        "action": "Choose lane according to desired direction."
    },
    37: {
        "name": "Go straight or left",
        "meaning": "Allowed to travel straight or turn left.",
        "action": "Choose lane according to desired direction."
    },
    38: {
        "name": "Keep right",
        "meaning": "Keep to the right of the obstacle or divider.",
        "action": "Pass the obstruction on the right side."
    },
    39: {
        "name": "Keep left",
        "meaning": "Keep to the left of the obstacle or divider.",
        "action": "Pass the obstruction on the left side."
    },
    40: {
        "name": "Roundabout mandatory",
        "meaning": "Roundabout ahead (Traffic inside has priority).",
        "action": "Yield to vehicles already inside the roundabout."
    },
    41: {
        "name": "End of no passing",
        "meaning": "End of overtaking prohibition zone.",
        "action": "Passing is now permitted if road conditions are safe."
    },
    42: {
        "name": "End of no passing by vehicles over 3.5 metric tons",
        "meaning": "End of truck overtaking restriction.",
        "action": "Heavy vehicles > 3.5t may now overtake safely."
    }
}
