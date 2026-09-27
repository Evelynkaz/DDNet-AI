(function () {
  "use strict";

  var loginView = document.getElementById("login-view");
  var statusView = document.getElementById("status-view");
  var loginForm = document.getElementById("login-form");
  var loginError = document.getElementById("login-error");
  var passwordInput = document.getElementById("password");
  var logoutButton = document.getElementById("logout-button");
  var connDot = document.getElementById("conn-dot");
  var wsStateEl = document.getElementById("ws-state");
  var botStateEl = document.getElementById("bot-state");
  var uptimeEl = document.getElementById("uptime");

  var csrfToken = null;
  var socket = null;

  function showLogin() {
    loginView.hidden = false;
    statusView.hidden = true;
    setConnected(false);
  }

  function showStatus() {
    loginView.hidden = true;
    statusView.hidden = false;
  }

  function setConnected(on) {
    connDot.classList.toggle("dot-on", on);
    connDot.classList.toggle("dot-off", !on);
    wsStateEl.textContent = on ? "подключено" : "не подключено";
  }

  function pad2(value) {
    return String(value).padStart(2, "0");
  }

  function formatUptime(seconds) {
    var total = Math.max(0, Math.floor(seconds));
    var h = Math.floor(total / 3600);
    var m = Math.floor((total % 3600) / 60);
    var s = total % 60;
    return pad2(h) + ":" + pad2(m) + ":" + pad2(s);
  }

  function connectWs() {
    if (socket) {
      return;
    }
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(proto + "//" + location.host + "/ws");
    socket.addEventListener("message", function (event) {
      var msg;
      try {
        msg = JSON.parse(event.data);
      } catch (parseError) {
        return;
      }
      // Connection state is confirmed once the app-level `hello` message arrives, which proves
      // both the WebSocket handshake and our own protocol worked, not merely `onopen`.
      if (msg.type === "hello") {
        setConnected(true);
      } else if (msg.type === "status") {
        botStateEl.textContent = msg.bot_state;
        uptimeEl.textContent = formatUptime(msg.uptime_s);
      }
    });
    socket.addEventListener("close", function () {
      setConnected(false);
      socket = null;
    });
    socket.addEventListener("error", function () {
      setConnected(false);
    });
  }

  function disconnectWs() {
    if (socket) {
      socket.close();
      socket = null;
    }
    setConnected(false);
  }

  function refresh() {
    fetch("/api/me", { credentials: "same-origin" })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        if (data.authenticated) {
          csrfToken = data.csrf_token;
          showStatus();
          connectWs();
        } else {
          csrfToken = null;
          showLogin();
        }
      })
      .catch(function () {
        showLogin();
      });
  }

  loginForm.addEventListener("submit", function (event) {
    event.preventDefault();
    loginError.hidden = true;
    fetch("/api/login", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: passwordInput.value }),
    })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("login failed");
        }
        return response.json();
      })
      .then(function (data) {
        csrfToken = data.csrf_token;
        passwordInput.value = "";
        showStatus();
        connectWs();
      })
      .catch(function () {
        loginError.textContent = "Неверный пароль или сервер недоступен.";
        loginError.hidden = false;
      });
  });

  logoutButton.addEventListener("click", function () {
    fetch("/api/logout", {
      method: "POST",
      credentials: "same-origin",
      headers: { "X-CSRF-Token": csrfToken || "" },
    }).finally(function () {
      disconnectWs();
      csrfToken = null;
      showLogin();
    });
  });

  refresh();
})();
