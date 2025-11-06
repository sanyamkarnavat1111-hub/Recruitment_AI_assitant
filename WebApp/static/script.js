let currentThreadId = null;
const threadList = document.getElementById("threadList");
const newChatBtn = document.getElementById("newChatBtn");
const chatMessages = document.getElementById("chatMessages");
const sendBtn = document.getElementById("sendBtn");
const userInput = document.getElementById("userInput");
const jobDescFile = document.getElementById("jobDescFile");
const jobDescUpload = document.getElementById("jobDescUpload");
const resumeFile = document.getElementById("resumeFile");

function createThread() {
  const threadId = "thread-" + Date.now();
  currentThreadId = threadId;

  const li = document.createElement("li");
  li.textContent = "Chat " + new Date().toLocaleTimeString();
  li.dataset.threadId = threadId;
  li.classList.add("active");

  document.querySelectorAll("#threadList li").forEach(item => item.classList.remove("active"));
  threadList.appendChild(li);
  li.addEventListener("click", () => switchThread(threadId, li));

  chatMessages.innerHTML = "";
}

function switchThread(threadId, li) {
  currentThreadId = threadId;
  document.querySelectorAll("#threadList li").forEach(item => item.classList.remove("active"));
  li.classList.add("active");
  chatMessages.innerHTML = `<div class="message bot"><div class="bubble">Switched to ${li.textContent}</div></div>`;
}

function addMessage(sender, text) {
  const msgDiv = document.createElement("div");
  msgDiv.classList.add("message", sender);
  const bubble = document.createElement("div");
  bubble.classList.add("bubble");
  bubble.textContent = text;
  msgDiv.appendChild(bubble);
  chatMessages.appendChild(msgDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

sendBtn.addEventListener("click", () => {
  const text = userInput.value.trim();
  if (!text) return;
  addMessage("user", text);
  userInput.value = "";
  setTimeout(() => addMessage("bot", "This is a placeholder response."), 800);
});

userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendBtn.click();
});

jobDescFile.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    addMessage("user", `Uploaded Job Description: ${file.name}`);
    // Simulate response
    setTimeout(() => {
      addMessage("bot", "Job description uploaded successfully.");
      jobDescUpload.style.display = "none";
    }, 1000);
  }
});

resumeFile.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    addMessage("user", `Uploaded Resume: ${file.name}`);
    setTimeout(() => addMessage("bot", "Resume uploaded successfully."), 800);
  }
});

newChatBtn.addEventListener("click", createThread);

// Initialize with one thread
createThread();
