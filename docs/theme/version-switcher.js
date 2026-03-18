// Version switcher for mdBook — reads versions.json and renders a dropdown.
// Auto-detects the base path from the URL so it works on any deployment
// (GitHub Pages project sites, custom domains, etc.) without configuration.
(function () {
  "use strict";

  // Detect base path and current version from the URL.
  // Expected URL structure: /<base>/<version>/<page>
  // e.g. /bloqade-lanes/dev/arch/archspec.html
  //      ^^^^^^^^^^^^^^^^ base   ^^^ version
  function detectPaths() {
    var path = window.location.pathname;

    // versions.json is in the same directory as the version — fetch it
    // relative to the current page's version root.
    // Walk up from the current page to the version root by finding
    // versions.json at each ancestor level.
    var segments = path.split("/").filter(Boolean);

    // Try each segment as a potential version identifier.
    // The version segment is followed by page content (index.html, arch/, etc.)
    for (var i = 0; i < segments.length; i++) {
      var candidate = segments[i];
      // Version segments look like "dev" or "vN.N.N"
      if (candidate === "dev" || /^v\d+/.test(candidate)) {
        // When version is the first segment (i == 0), basePath should be
        // empty string, not "/" — otherwise URL construction produces "//".
        var basePath = i > 0 ? "/" + segments.slice(0, i).join("/") : "";
        return {
          basePath: basePath,
          currentVersion: candidate,
          versionsUrl: basePath + "/" + candidate + "/versions.json"
        };
      }
    }

    // Fallback: assume no base path
    return {
      basePath: "",
      currentVersion: "dev",
      versionsUrl: "/versions.json"
    };
  }

  var paths = detectPaths();

  fetch(paths.versionsUrl)
    .then(function (res) {
      if (!res.ok) {
        console.warn(
          "[version-switcher] Failed to fetch " + paths.versionsUrl +
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

      // Build the dropdown
      var select = document.createElement("select");
      select.id = "version-switcher";
      select.style.cssText =
        "margin-left: 1rem; padding: 0.25rem 0.5rem; border-radius: 4px; " +
        "border: 1px solid var(--sidebar-fg); background: var(--sidebar-bg); " +
        "color: var(--sidebar-fg); font-size: 0.85rem; cursor: pointer;";

      for (var j = 0; j < versions.length; j++) {
        var ver = versions[j].version;
        var opt = document.createElement("option");
        // Construct URL from detected base path + version label
        opt.value = paths.basePath + "/" + ver + "/";
        opt.textContent = ver;
        if (ver === paths.currentVersion) {
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
