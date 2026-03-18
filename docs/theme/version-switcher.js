// Version switcher for mdBook — reads /versions.json and renders a dropdown.
(function () {
  "use strict";

  // Resolve the base URL (handles both local preview and GitHub Pages)
  var scriptEl = document.currentScript;
  var baseUrl = scriptEl
    ? scriptEl.getAttribute("data-base-url") || ""
    : "";

  var versionsUrl = baseUrl + "/versions.json";

  fetch(versionsUrl)
    .then(function (res) {
      if (!res.ok) {
        console.warn(
          "[version-switcher] Failed to fetch " + versionsUrl +
          " (HTTP " + res.status + "). Version switcher disabled."
        );
        return null;
      }
      return res.json();
    })
    .then(function (versions) {
      if (!versions) return;

      if (!Array.isArray(versions) || versions.length === 0) {
        console.warn(
          "[version-switcher] versions.json is empty or not an array. " +
          "Version switcher disabled."
        );
        return;
      }

      // Detect the current version from the URL path
      var path = window.location.pathname;
      var currentVersion = "dev";
      for (var i = 0; i < versions.length; i++) {
        if (path.indexOf("/" + versions[i].version + "/") !== -1) {
          currentVersion = versions[i].version;
          break;
        }
      }

      // Build the dropdown
      var select = document.createElement("select");
      select.id = "version-switcher";
      select.style.cssText =
        "margin-left: 1rem; padding: 0.25rem 0.5rem; border-radius: 4px; " +
        "border: 1px solid var(--sidebar-fg); background: var(--sidebar-bg); " +
        "color: var(--sidebar-fg); font-size: 0.85rem; cursor: pointer;";

      for (var j = 0; j < versions.length; j++) {
        var opt = document.createElement("option");
        opt.value = versions[j].url;
        opt.textContent = versions[j].version;
        if (versions[j].version === currentVersion) {
          opt.selected = true;
        }
        select.appendChild(opt);
      }

      select.addEventListener("change", function () {
        window.location.href = this.value;
      });

      // Insert into the mdBook menu bar
      var menuBar = document.querySelector(".right-buttons");
      if (!menuBar) {
        console.warn(
          "[version-switcher] Could not find .right-buttons element. " +
          "Version dropdown not rendered."
        );
        return;
      }

      var wrapper = document.createElement("div");
      wrapper.style.cssText = "display: inline-flex; align-items: center;";

      var label = document.createElement("span");
      label.textContent = "Version: ";
      label.style.cssText =
        "font-size: 0.85rem; color: var(--sidebar-fg); margin-right: 0.25rem;";

      wrapper.appendChild(label);
      wrapper.appendChild(select);
      menuBar.prepend(wrapper);
    })
    .catch(function (err) {
      console.warn(
        "[version-switcher] Error loading version switcher: " + err.message
      );
    });
})();
