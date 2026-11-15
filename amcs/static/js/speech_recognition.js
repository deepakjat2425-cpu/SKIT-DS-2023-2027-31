/**
 * Hands-Free Voice Commands Module using Web Speech Recognition API
 * Allows visually impaired users to control the system via spoken voice commands.
 * Automatic Multilanguage Communication System - SKIT Jaipur
 */

let speechRecognition = null;
let isListening = false;

function initVoiceRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn("[AMCS] Web Speech Recognition not supported in this browser.");
        return null;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onstart = () => {
        isListening = true;
        updateMicButtonUI(true);
        console.log("[AMCS Voice] Listening for voice commands...");
    };

    recognition.onresult = (event) => {
        const lastResult = event.results[event.results.length - 1];
        if (lastResult.isFinal) {
            const transcript = lastResult[0].transcript.trim().toLowerCase();
            console.log(`[AMCS Voice] Detected command: "${transcript}"`);
            handleVoiceCommand(transcript);
        }
    };

    recognition.onerror = (event) => {
        console.warn("[AMCS Voice] Recognition error:", event.error);
        if (event.error === "not-allowed") {
            alert("Microphone permission denied. Please allow microphone access for hands-free voice commands.");
        }
    };

    recognition.onend = () => {
        if (isListening) {
            // Keep active
            try { recognition.start(); } catch (e) {}
        } else {
            updateMicButtonUI(false);
        }
    };

    return recognition;
}

function toggleVoiceRecognition() {
    if (!speechRecognition) {
        speechRecognition = initVoiceRecognition();
    }

    if (!speechRecognition) {
        alert("Voice recognition is not supported in your browser (Chrome/Edge recommended).");
        return;
    }

    if (isListening) {
        isListening = false;
        speechRecognition.stop();
        updateMicButtonUI(false);
        speakViaBrowser("Voice command mode deactivated.", "en");
    } else {
        isListening = true;
        try {
            speechRecognition.start();
            speakViaBrowser("Voice command mode active. You can say: What is ahead, Emergency, Read sign, or Stop.", "en");
        } catch (e) {
            console.warn(e);
        }
    }
}

function updateMicButtonUI(active) {
    const btn = document.getElementById("mic-btn");
    const txt = document.getElementById("mic-text");
    if (!btn) return;

    if (active) {
        btn.classList.add("listening");
        if (txt) txt.textContent = "Listening...";
    } else {
        btn.classList.remove("listening");
        if (txt) txt.textContent = "Voice Mic";
    }
}

function handleVoiceCommand(command) {
    // 1. Scene description query
    if (command.includes("what is ahead") || command.includes("describe") || command.includes("look") || command.includes("what do you see")) {
        speakViaBrowser("Analyzing surroundings now.", langSelect ? langSelect.value : "en");
        if (typeof captureManualFrame === "function") {
            captureManualFrame();
        }
    }
    // 2. Emergency SOS
    else if (command.includes("emergency") || command.includes("help") || command.includes("sos") || command.includes("danger")) {
        if (typeof triggerSOS === "function") {
            triggerSOS();
        }
    }
    // 3. Start / Stop Automation
    else if (command.includes("start") || command.includes("resume") || command.includes("begin")) {
        if (!isAutoRunning && typeof toggleAutoDetection === "function") {
            toggleAutoDetection();
        }
    }
    else if (command.includes("stop") || command.includes("pause") || command.includes("halt")) {
        if (isAutoRunning && typeof toggleAutoDetection === "function") {
            toggleAutoDetection();
        }
    }
    // 4. Switch to Signboard / OCR
    else if (command.includes("read sign") || command.includes("text") || command.includes("signboard") || command.includes("read board")) {
        if (typeof setMode === "function") {
            setMode("ocr");
            speakViaBrowser("Switched to Signboard reading mode.", "en");
        }
    }
    // 5. Switch to Obstacle mode
    else if (command.includes("obstacle") || command.includes("navigation") || command.includes("walk")) {
        if (typeof setMode === "function") {
            setMode("vision");
            speakViaBrowser("Switched to Obstacle Navigation mode.", "en");
        }
    }
    // 6. Language switching
    else if (command.includes("hindi")) {
        setLanguageCode("hi");
    }
    else if (command.includes("english")) {
        setLanguageCode("en");
    }
    else if (command.includes("spanish")) {
        setLanguageCode("es");
    }
    else if (command.includes("french")) {
        setLanguageCode("fr");
    }
    else if (command.includes("german")) {
        setLanguageCode("de");
    }
    else {
        console.log(`[AMCS Voice] Unrecognized command: "${command}"`);
    }
}

function setLanguageCode(code) {
    if (langSelect) {
        langSelect.value = code;
        langSelect.dispatchEvent(new Event("change"));
    }
}
