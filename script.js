const chatWindow = document.getElementById("chat-window");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");

// Add user message
function appendMessage(sender, text) {
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message");
  messageDiv.classList.add(sender === "user" ? "user-message" : "aether-message");
  messageDiv.textContent = text;
  chatWindow.appendChild(messageDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Handle send
async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  appendMessage("user", message);
  userInput.value = "";

  // Optional: Typing indicator
  const typing = document.createElement("div");
  typing.classList.add("message", "aether-message");
  typing.textContent = "Aether is thinking...";
  chatWindow.appendChild(typing);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  try {
    const res = await fetch("http://localhost:8000/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ message })
    });

    const data = await res.json();
    chatWindow.removeChild(typing);
    appendMessage("aether", data.response);
  } catch (error) {
    chatWindow.removeChild(typing);
    appendMessage("aether", "Oops, something went wrong.");
    console.error("Error:", error);
  }
}

// Send on button click or Enter key
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});

