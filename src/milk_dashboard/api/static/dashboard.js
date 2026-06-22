const state = {
  settings: {
    horizonDays: 7,
    lookbackDays: 30,
    forecastModel: "mlflow_pretrained",
    safetyBufferPct: 10,
    showForecastRange: true,
    splitForecastCharts: false,
    weekdayMode: "mean",
    topNItems: 12,
  },
  salesFilters: {
    windowMode: "all_time",
    startDate: "",
    endDate: "",
    drillCategory: "",
  },
  forecast: null,
  sales: null,
  ingredients: [],
  ingredientSearch: {
    query: "",
    matches: [],
    activeIndex: -1,
  },
  admin: {
    modelConfig: null,
    latestEod: null,
  },
  auth: {
    me: null,
  },
};

const el = {
  appShell: document.querySelector(".app-shell"),
  sidebarToggleBtn: document.getElementById("sidebarToggleBtn"),
  authEmail: document.getElementById("authEmail"),
  authPassword: document.getElementById("authPassword"),
  authEmailError: document.getElementById("authEmailError"),
  authPasswordError: document.getElementById("authPasswordError"),
  authRegisterBtn: document.getElementById("authRegisterBtn"),
  authLoginBtn: document.getElementById("authLoginBtn"),
  authLogoutBtn: document.getElementById("authLogoutBtn"),
  squareConnectBtn: document.getElementById("squareConnectBtn"),
  squareDisconnectBtn: document.getElementById("squareDisconnectBtn"),
  authStatus: document.getElementById("authStatus"),
  userBadge: document.getElementById("userBadge"),
  userInitials: document.getElementById("userInitials"),
  userBadgeText: document.getElementById("userBadgeText"),
  registerModal: document.getElementById("registerModal"),
  registerModalCloseBtn: document.getElementById("registerModalCloseBtn"),
  registerCancelBtn: document.getElementById("registerCancelBtn"),
  registerSubmitBtn: document.getElementById("registerSubmitBtn"),
  registerFirstName: document.getElementById("registerFirstName"),
  registerLastName: document.getElementById("registerLastName"),
  registerEmail: document.getElementById("registerEmail"),
  registerPassword: document.getElementById("registerPassword"),
  registerFirstNameError: document.getElementById("registerFirstNameError"),
  registerLastNameError: document.getElementById("registerLastNameError"),
  registerEmailError: document.getElementById("registerEmailError"),
  registerPasswordError: document.getElementById("registerPasswordError"),
  registerStatus: document.getElementById("registerStatus"),
  horizonDays: document.getElementById("horizonDays"),
  lookbackDays: document.getElementById("lookbackDays"),
  forecastModel: document.getElementById("forecastModel"),
  safetyBufferPct: document.getElementById("safetyBufferPct"),
  showForecastRange: document.getElementById("showForecastRange"),
  splitForecastCharts: document.getElementById("splitForecastCharts"),
  weekdayMode: document.getElementById("weekdayMode"),
  topNItems: document.getElementById("topNItems"),
  themeSelect: document.getElementById("themeSelect"),
  applySettingsBtn: document.getElementById("applySettingsBtn"),
  refreshDataBtn: document.getElementById("refreshDataBtn"),
  addIngredientRowBtn: document.getElementById("addIngredientRowBtn"),
  saveIngredientBtn: document.getElementById("saveIngredientBtn"),
  reloadIngredientBtn: document.getElementById("reloadIngredientBtn"),
  ingredientSearch: document.getElementById("ingredientSearch"),
  ingredientSearchBtn: document.getElementById("ingredientSearchBtn"),
  ingredientSearchStatus: document.getElementById("ingredientSearchStatus"),
  applySalesWindowBtn: document.getElementById("applySalesWindowBtn"),
  salesWindowMode: document.getElementById("salesWindowMode"),
  salesStartDate: document.getElementById("salesStartDate"),
  salesEndDate: document.getElementById("salesEndDate"),
  salesStartWrap: document.getElementById("salesStartWrap"),
  salesEndWrap: document.getElementById("salesEndWrap"),
  salesCategorySelect: document.getElementById("salesCategorySelect"),
  sidebarStatus: document.getElementById("sidebarStatus"),
  ingredientStatus: document.getElementById("ingredientStatus"),
  salesWindowStatus: document.getElementById("salesWindowStatus"),
  ingredientFileName: document.getElementById("ingredientFileName"),
  ingredientTableBody: document.getElementById("ingredientTableBody"),
  forecastTablesWrap: document.getElementById("forecastTablesWrap"),
  patternTablesWrap: document.getElementById("patternTablesWrap"),
  salesTablesWrap: document.getElementById("salesTablesWrap"),
  adminApiKey: document.getElementById("adminApiKey"),
  adminModelName: document.getElementById("adminModelName"),
  adminModelAlias: document.getElementById("adminModelAlias"),
  adminLoadConfigBtn: document.getElementById("adminLoadConfigBtn"),
  adminSaveConfigBtn: document.getElementById("adminSaveConfigBtn"),
  adminModelStatus: document.getElementById("adminModelStatus"),
  adminMerchantId: document.getElementById("adminMerchantId"),
  adminDryRun: document.getElementById("adminDryRun"),
  adminRunEodBtn: document.getElementById("adminRunEodBtn"),
  adminLoadLatestEodBtn: document.getElementById("adminLoadLatestEodBtn"),
  adminEodStatus: document.getElementById("adminEodStatus"),
  adminEodSummary: document.getElementById("adminEodSummary"),
  horizonDaysValue: document.getElementById("horizonDaysValue"),
  lookbackDaysValue: document.getElementById("lookbackDaysValue"),
  safetyBufferPctValue: document.getElementById("safetyBufferPctValue"),
  topNItemsValue: document.getElementById("topNItemsValue"),
};

function setStatus(target, message, tone = "") {
  if (!target) return;
  target.textContent = message;
  target.classList.remove("error", "success");
  if (tone) {
    target.classList.add(tone);
  }
}

function setFieldError(input, errorNode, message = "") {
  if (errorNode) {
    errorNode.textContent = message;
  }
  if (input) {
    input.classList.toggle("input-error", Boolean(message));
    input.setAttribute("aria-invalid", message ? "true" : "false");
  }
}

function requireField(input, errorNode, label) {
  const value = (input?.value || "").trim();
  if (!value) {
    setFieldError(input, errorNode, `${label} is required.`);
    return false;
  }
  setFieldError(input, errorNode);
  return true;
}

function isValidEmail(value) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
}

function requireEmail(input, errorNode) {
  const value = (input?.value || "").trim();
  if (!value) {
    setFieldError(input, errorNode, "Email is required.");
    return false;
  }
  if (!isValidEmail(value)) {
    setFieldError(input, errorNode, "Enter a valid email address.");
    return false;
  }
  setFieldError(input, errorNode);
  return true;
}

function requirePassword(input, errorNode) {
  const value = (input?.value || "").trim();
  if (!value) {
    setFieldError(input, errorNode, "Password is required.");
    return false;
  }
  if (value.length < 8) {
    setFieldError(input, errorNode, "Password must be at least 8 characters.");
    return false;
  }
  setFieldError(input, errorNode);
  return true;
}

function toNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function formatNum(value, digits = 2) {
  return toNumber(value).toFixed(digits);
}

function formatCurrency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(toNumber(value));
}

function palette() {
  const dark = document.body.getAttribute("data-theme") === "dark";
  const salesPalette = ["#60a5fa", "#22d3ee", "#34d399", "#818cf8", "#2dd4bf", "#38bdf8", "#10b981", "#06b6d4", "#4ade80", "#93c5fd"];
  if (dark) {
    return {
      qty: "#2f5f8f",
      sales: "#f8fafc",
      past: "#f8fafc",
      forecast: "#f8fafc",
      range: "rgba(248, 250, 252, 0.12)",
      barOpacity: 1,
      weekday: "#2f5f8f",
      monthBar: "#2f5f8f",
      monthLine: "#f8fafc",
      salesLine: "#f8fafc",
      salesBars: salesPalette,
      comp: salesPalette,
      drillBars: salesPalette,
      drillLine: "#f8fafc",
    };
  }
  return {
    qty: "#2f5f8f",
    sales: "#0f172a",
    past: "#0f172a",
    forecast: "#0f172a",
    range: "rgba(15, 23, 42, 0.1)",
    barOpacity: 1,
    weekday: "#2f5f8f",
    monthBar: "#2f5f8f",
    monthLine: "#0f172a",
    salesLine: "#0f172a",
    salesBars: ["#2563eb", "#0891b2", "#059669", "#4f46e5", "#0f766e", "#0284c7", "#047857", "#0e7490", "#16a34a", "#1d4ed8"],
    comp: ["#2563eb", "#0891b2", "#059669", "#4f46e5", "#0f766e", "#0284c7", "#047857", "#0e7490", "#16a34a", "#1d4ed8"],
    drillBars: ["#2563eb", "#0891b2", "#059669", "#4f46e5", "#0f766e", "#0284c7", "#047857", "#0e7490", "#16a34a", "#1d4ed8"],
    drillLine: "#0f172a",
  };
}

function hexToRgba(hex, alpha) {
  if (!hex || typeof hex !== "string") return `rgba(96, 165, 250, ${alpha})`;
  const clean = hex.replace("#", "");
  if (!(clean.length === 6 || clean.length === 3)) return `rgba(96, 165, 250, ${alpha})`;
  const full = clean.length === 3 ? clean.split("").map((c) => `${c}${c}`).join("") : clean;
  const intVal = Number.parseInt(full, 16);
  const r = (intVal >> 16) & 255;
  const g = (intVal >> 8) & 255;
  const b = intVal & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    if (typeof payload === "object" && payload?.detail) {
      const detail = payload.detail;
      if (Array.isArray(detail)) {
        const messages = detail
          .map((item) => {
            if (typeof item === "string") return item;
            const field = Array.isArray(item?.loc) ? item.loc[item.loc.length - 1] : null;
            const msg = item?.msg || "Invalid value";
            return field ? `${field}: ${msg}` : msg;
          })
          .join(", ");
        throw new Error(messages || "Request failed validation.");
      }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    throw new Error(typeof payload === "string" ? payload : `Request failed: ${response.status}`);
  }
  return payload;
}

function adminHeaders() {
  const headers = {};
  const key = (el.adminApiKey?.value || "").trim();
  if (key) {
    headers["x-admin-key"] = key;
  }
  return headers;
}

function loginPayloadFromInputs() {
  return {
    email: (el.authEmail?.value || "").trim(),
    password: (el.authPassword?.value || "").trim(),
  };
}

function registerPayloadFromInputs() {
  return {
    first_name: (el.registerFirstName?.value || "").trim(),
    last_name: (el.registerLastName?.value || "").trim(),
    email: (el.registerEmail?.value || "").trim(),
    password: (el.registerPassword?.value || "").trim(),
  };
}

function validateLoginInputs() {
  return [
    requireEmail(el.authEmail, el.authEmailError),
    requirePassword(el.authPassword, el.authPasswordError),
  ].every(Boolean);
}

function validateRegisterInputs() {
  return [
    requireField(el.registerFirstName, el.registerFirstNameError, "First name"),
    requireField(el.registerLastName, el.registerLastNameError, "Last name"),
    requireEmail(el.registerEmail, el.registerEmailError),
    requirePassword(el.registerPassword, el.registerPasswordError),
  ].every(Boolean);
}

function userDisplayName(user) {
  const name = [user?.first_name, user?.last_name].filter(Boolean).join(" ").trim();
  return name || user?.email || "Signed in";
}

function userInitials(user) {
  const first = (user?.first_name || "").trim();
  const last = (user?.last_name || "").trim();
  if (first || last) {
    return `${first.charAt(0)}${last.charAt(0)}`.toUpperCase();
  }
  return (user?.email || "?").charAt(0).toUpperCase();
}

function updateAuthUi() {
  const me = state.auth.me;
  const signedIn = Boolean(me?.authenticated);
  el.userBadge?.classList.toggle("hidden", !signedIn);
  if (signedIn) {
    el.userInitials.textContent = userInitials(me.user);
    el.userBadgeText.textContent = userDisplayName(me.user);
  } else {
    el.userInitials.textContent = "--";
    el.userBadgeText.textContent = "Signed out";
  }
  if (el.authLogoutBtn) el.authLogoutBtn.disabled = !signedIn;
  if (el.squareConnectBtn) el.squareConnectBtn.disabled = !signedIn;
  if (el.squareDisconnectBtn) el.squareDisconnectBtn.disabled = !signedIn || !me?.square_connection;
}

function openRegisterModal() {
  setStatus(el.registerStatus, "");
  [el.registerFirstName, el.registerLastName, el.registerEmail, el.registerPassword].forEach((input) => {
    if (input) input.value = "";
  });
  [
    [el.registerFirstName, el.registerFirstNameError],
    [el.registerLastName, el.registerLastNameError],
    [el.registerEmail, el.registerEmailError],
    [el.registerPassword, el.registerPasswordError],
  ].forEach(([input, errorNode]) => setFieldError(input, errorNode));
  el.registerModal?.classList.remove("hidden");
  el.registerFirstName?.focus();
}

function closeRegisterModal() {
  el.registerModal?.classList.add("hidden");
}

async function loadAuthMe() {
  state.auth.me = await fetchJSON("/api/auth/me");
  const me = state.auth.me;
  if (!me.authenticated) {
    setStatus(el.authStatus, "Not logged in.");
    updateAuthUi();
    return;
  }
  const merchantId = me.square_connection?.merchant_id || "not connected";
  setStatus(el.authStatus, `Logged in as ${userDisplayName(me.user)}. Square: ${merchantId}.`, "success");
  updateAuthUi();
}

async function registerAuth() {
  if (!validateRegisterInputs()) {
    setStatus(el.registerStatus, "Please complete the required fields.", "error");
    return;
  }
  const payload = registerPayloadFromInputs();
  el.registerSubmitBtn.disabled = true;
  try {
    await fetchJSON("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await loadAuthMe();
    closeRegisterModal();
    setStatus(el.authStatus, `Account created. Logged in as ${payload.first_name} ${payload.last_name}.`, "success");
  } finally {
    el.registerSubmitBtn.disabled = false;
  }
}

async function loginAuth() {
  if (!validateLoginInputs()) {
    setStatus(el.authStatus, "Please enter your email and password.", "error");
    return;
  }
  const payload = loginPayloadFromInputs();
  await fetchJSON("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await loadAuthMe();
}

async function logoutAuth() {
  await fetchJSON("/api/auth/logout", { method: "POST" });
  await loadAuthMe();
  setStatus(el.authStatus, "Logged out.");
}

function syncSettingLabels() {
  el.horizonDaysValue.textContent = state.settings.horizonDays;
  el.lookbackDaysValue.textContent = "All history";
  el.safetyBufferPctValue.textContent = state.settings.safetyBufferPct;
  el.topNItemsValue.textContent = state.settings.topNItems;
}

function readSettingsFromInputs() {
  state.settings.horizonDays = toNumber(el.horizonDays.value, 7);
  state.settings.lookbackDays = 30;
  state.settings.forecastModel = el.forecastModel.value || "mlflow_pretrained";
  state.settings.safetyBufferPct = toNumber(el.safetyBufferPct.value, 10);
  state.settings.showForecastRange = Boolean(el.showForecastRange.checked);
  state.settings.splitForecastCharts = Boolean(el.splitForecastCharts.checked);
  state.settings.weekdayMode = el.weekdayMode.value;
  state.settings.topNItems = toNumber(el.topNItems.value, 12);
  syncSettingLabels();
}

function readSalesFiltersFromInputs() {
  state.salesFilters.windowMode = el.salesWindowMode.value;
  state.salesFilters.startDate = (el.salesStartDate.value || "").trim();
  state.salesFilters.endDate = (el.salesEndDate.value || "").trim();
  state.salesFilters.drillCategory = (el.salesCategorySelect.value || "").trim();
}

function toggleCustomDateInputs() {
  const isCustom = el.salesWindowMode.value === "custom";
  el.salesStartWrap.classList.toggle("hidden", !isCustom);
  el.salesEndWrap.classList.toggle("hidden", !isCustom);
}

function applyTheme(theme) {
  document.body.setAttribute("data-theme", theme);
  if (state.forecast) renderForecast();
  if (state.sales) renderSales();
}

function plotLayout(title = "") {
  const isDark = document.body.getAttribute("data-theme") === "dark";
  const color = getComputedStyle(document.body).getPropertyValue("--text").trim();
  const grid = isDark ? "rgba(182, 197, 214, 0.2)" : "rgba(39, 65, 102, 0.2)";
  const bg = "rgba(0,0,0,0)";
  const hoverBg = isDark ? "rgba(15, 23, 36, 0.98)" : "rgba(247, 250, 253, 0.98)";
  const hoverText = isDark ? "#f8fafc" : "#122033";
  const hoverBorder = isDark ? "#2f9f85" : "#2f6f98";
  return {
    title: title ? { text: title, font: { size: 13 } } : undefined,
    paper_bgcolor: bg,
    plot_bgcolor: bg,
    font: { color, family: "Space Grotesk, sans-serif" },
    margin: { l: 58, r: 50, t: 26, b: 110 },
    xaxis: { gridcolor: grid, zeroline: false, automargin: true },
    yaxis: { gridcolor: grid, zeroline: false, automargin: true },
    legend: {
      orientation: "h",
      x: 0,
      xanchor: "left",
      y: -0.2,
      yanchor: "top",
      bgcolor: "rgba(0,0,0,0)",
    },
    hoverlabel: {
      bgcolor: hoverBg,
      bordercolor: hoverBorder,
      font: { color: hoverText, size: 12, family: "Space Grotesk, sans-serif" },
      align: "left",
    },
    hovermode: "x unified",
  };
}

function resizeVisibleCharts() {
  const chartIds = [
    "combinedForecastChart",
    "pastOnlyChart",
    "forecastOnlyChart",
    "weekdayPatternChart",
    "monthlyPatternChart",
    "categoryDualAxisChart",
    "categoryCompositionChart",
    "itemDrilldownChart",
  ];
  chartIds.forEach((id) => {
    const node = document.getElementById(id);
    if (!node || node.offsetParent === null) return;
    Plotly.Plots.resize(node);
  });
}

function renderTable(title, rows, columns) {
  const safeRows = Array.isArray(rows) ? rows : [];
  if (!safeRows.length) {
    return `
      <article class="card">
        <div class="card-head"><h3>${title}</h3></div>
        <p class="muted">No rows available.</p>
      </article>
    `;
  }
  const header = columns.map((c) => `<th>${c.label}</th>`).join("");
  const body = safeRows
    .map((row) => {
      const cells = columns.map((c) => `<td>${String(row[c.key] ?? "")}</td>`).join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
  return `
    <article class="card">
      <div class="card-head"><h3>${title}</h3></div>
      <div class="table-wrap">
        <table>
          <thead><tr>${header}</tr></thead>
          <tbody>${body}</tbody>
        </table>
      </div>
    </article>
  `;
}

function clearIngredientSearchHighlights() {
  el.ingredientTableBody
    .querySelectorAll("tr.ingredient-match")
    .forEach((row) => row.classList.remove("ingredient-match"));
}

function resetIngredientSearchState() {
  state.ingredientSearch.query = "";
  state.ingredientSearch.matches = [];
  state.ingredientSearch.activeIndex = -1;
  clearIngredientSearchHighlights();
  if (el.ingredientSearchStatus) {
    setStatus(el.ingredientSearchStatus, "");
  }
}

function runIngredientSearch() {
  const query = (el.ingredientSearch.value || "").trim().toLowerCase();
  if (!query) {
    resetIngredientSearchState();
    setStatus(el.ingredientSearchStatus, "Type text to find category, item, or price point.");
    return;
  }

  const rows = [...el.ingredientTableBody.querySelectorAll("tr")];
  const rowMatches = rows.filter((tr) => {
    const text = [...tr.querySelectorAll("input")]
      .map((input) => (input.value || "").toLowerCase())
      .join(" ");
    return text.includes(query);
  });

  if (!rowMatches.length) {
    resetIngredientSearchState();
    state.ingredientSearch.query = query;
    setStatus(el.ingredientSearchStatus, `No matches for "${query}".`, "error");
    return;
  }

  if (state.ingredientSearch.query === query && state.ingredientSearch.matches.length) {
    state.ingredientSearch.activeIndex = (state.ingredientSearch.activeIndex + 1) % state.ingredientSearch.matches.length;
  } else {
    state.ingredientSearch.query = query;
    state.ingredientSearch.matches = rowMatches;
    state.ingredientSearch.activeIndex = 0;
  }

  clearIngredientSearchHighlights();
  const activeRow = state.ingredientSearch.matches[state.ingredientSearch.activeIndex];
  activeRow.classList.add("ingredient-match");
  activeRow.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
  activeRow.querySelector("input")?.focus();

  const rowIndex = rows.indexOf(activeRow) + 1;
  setStatus(
    el.ingredientSearchStatus,
    `Match ${state.ingredientSearch.activeIndex + 1}/${state.ingredientSearch.matches.length} at row ${rowIndex}.`,
    "success"
  );
}

function weekdayLabel(dateValue) {
  if (!dateValue) return "";
  const dt = new Date(`${dateValue}T00:00:00`);
  if (Number.isNaN(dt.getTime())) return "";
  return dt.toLocaleDateString("en-US", { weekday: "long" });
}

function computeSharedYRange(seriesGroups) {
  const values = seriesGroups
    .flat()
    .map((v) => toNumber(v, NaN))
    .filter((v) => Number.isFinite(v));
  if (!values.length) return null;

  const minY = Math.min(...values);
  const maxY = Math.max(...values);
  if (minY === maxY) {
    const pad = Math.max(1, Math.abs(maxY) * 0.15);
    return [Math.max(0, minY - pad), maxY + pad];
  }
  const pad = (maxY - minY) * 0.1;
  return [Math.max(0, minY - pad), maxY + pad];
}

async function loadForecast() {
  const params = new URLSearchParams({
    horizon_days: String(state.settings.horizonDays),
    lookback_days: String(state.settings.lookbackDays),
    model: state.settings.forecastModel,
    safety_buffer_pct: String(state.settings.safetyBufferPct),
  });
  state.forecast = await fetchJSON(`/api/dashboard/forecast?${params.toString()}`);
  renderForecast();
}

async function loadSales() {
  readSalesFiltersFromInputs();
  const params = new URLSearchParams({
    top_n: String(state.settings.topNItems),
    window: state.salesFilters.windowMode,
  });
  if (state.salesFilters.windowMode === "custom") {
    if (state.salesFilters.startDate) params.set("start_date", state.salesFilters.startDate);
    if (state.salesFilters.endDate) params.set("end_date", state.salesFilters.endDate);
  }
  if (state.salesFilters.drillCategory) {
    params.set("drill_category", state.salesFilters.drillCategory);
  }
  state.sales = await fetchJSON(`/api/dashboard/sales-breakdown?${params.toString()}`);
  renderSales();
}

async function loadIngredients() {
  const payload = await fetchJSON("/api/dashboard/ingredients");
  state.ingredients = payload.rows || [];
  el.ingredientFileName.textContent = payload.file_name || "--";
  renderIngredientsTable();
}

function renderAdminEodSummary(report) {
  if (!el.adminEodSummary) return;
  if (!report) {
    el.adminEodSummary.innerHTML = "";
    return;
  }
  const sync = report.sync ? report.sync : report;
  const alerts = sync.alerts || report.alerts || {};
  const priorityAlerts = alerts.priority_alerts || alerts.coffee_alerts || [];
  const nonPriority = alerts.non_priority_new_items || alerts.non_coffee_new_items || [];

  el.adminEodSummary.innerHTML =
    renderTable("EOD Sync Summary", [
      {
        run_id: sync.run_id ?? "--",
        merchant_id: sync.merchant_id || "--",
        fetched_orders: sync.fetched_orders ?? 0,
        fetched_rows: sync.fetched_rows ?? 0,
        appended_rows: sync.appended_rows ?? 0,
        total_rows_after: sync.total_rows_after ?? 0,
        dry_run: Boolean(sync.dry_run),
      },
    ], [
      { key: "run_id", label: "Run ID" },
      { key: "merchant_id", label: "Merchant" },
      { key: "fetched_orders", label: "Orders" },
      { key: "fetched_rows", label: "Rows Pulled" },
      { key: "appended_rows", label: "Rows Appended" },
      { key: "total_rows_after", label: "CSV Rows After" },
      { key: "dry_run", label: "Dry Run" },
    ]) +
    renderTable("Priority Alerts (Ingredient Categories)", priorityAlerts, [
      { key: "category", label: "Category" },
      { key: "item", label: "Item" },
      { key: "price_point_name", label: "Price Point" },
      { key: "total_qty", label: "Qty" },
      { key: "suggested_action", label: "Action" },
    ]) +
    renderTable("New Non-Priority Items", nonPriority, [
      { key: "category", label: "Category" },
      { key: "item", label: "Item" },
      { key: "price_point_name", label: "Price Point" },
      { key: "total_qty", label: "Qty" },
      { key: "suggested_action", label: "Action" },
    ]);
}

async function loadAdminModelConfig() {
  const payload = await fetchJSON("/api/admin/model-config", {
    headers: adminHeaders(),
  });
  state.admin.modelConfig = payload;
  if (el.adminModelName) el.adminModelName.value = payload.model_name || "";
  if (el.adminModelAlias) el.adminModelAlias.value = payload.model_alias || "";
  setStatus(
    el.adminModelStatus,
    `Loaded model ${payload.model_name || "--"} @ ${payload.model_alias || "--"}`,
    "success"
  );
}

async function saveAdminModelConfig() {
  const modelName = (el.adminModelName?.value || "").trim();
  const modelAlias = (el.adminModelAlias?.value || "").trim();
  if (!modelName || !modelAlias) {
    throw new Error("Model name and alias are required.");
  }
  const payload = await fetchJSON("/api/admin/model-config", {
    method: "PUT",
    headers: { "Content-Type": "application/json", ...adminHeaders() },
    body: JSON.stringify({
      model_name: modelName,
      model_alias: modelAlias,
    }),
  });
  state.admin.modelConfig = payload.model_config || null;
  setStatus(
    el.adminModelStatus,
    `Saved model config to ${payload.runtime_config_path}`,
    "success"
  );
}

async function loadLatestEodSync() {
  const payload = await fetchJSON("/api/admin/eod/latest", {
    headers: adminHeaders(),
  });
  state.admin.latestEod = payload.latest || null;
  if (!payload.latest) {
    setStatus(el.adminEodStatus, "No EOD sync run found yet.");
    renderAdminEodSummary(null);
    return;
  }
  setStatus(
    el.adminEodStatus,
    `Loaded latest sync #${payload.latest.run_id} (${payload.latest.run_ts_utc})`,
    "success"
  );
  renderAdminEodSummary(payload.latest);
}

async function runEodSync() {
  const merchantId = (el.adminMerchantId?.value || "").trim();
  const dryRun = Boolean(el.adminDryRun?.checked);
  const payload = await fetchJSON("/api/admin/eod-sync", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...adminHeaders() },
    body: JSON.stringify({
      merchant_id: merchantId || null,
      dry_run: dryRun,
      horizon_days: state.settings.horizonDays,
      lookback_days: state.settings.lookbackDays,
      safety_buffer_pct: state.settings.safetyBufferPct,
    }),
  });
  state.admin.latestEod = payload.sync || null;
  setStatus(
    el.adminEodStatus,
    `EOD sync complete. Appended ${payload.sync?.appended_rows ?? 0} rows.`,
    "success"
  );
  renderAdminEodSummary(payload.sync);
  await runFullRefresh();
}

function renderForecast() {
  const payload = state.forecast;
  if (!payload) return;
  const colors = palette();
  const isDark = document.body.getAttribute("data-theme") === "dark";
  const markerOutline = isDark ? "#0b1220" : "#ffffff";
  const forecastMarker = {
    size: 7,
    symbol: "circle",
    color: colors.forecast,
    line: { width: 1.4, color: markerOutline },
  };

  const summary = payload.summary || {};
  const past = payload.series?.past || [];
  const forecast = payload.series?.forecast || [];
  const weekday = payload.patterns?.weekday || {};
  const monthly = payload.patterns?.monthly || [];

  document.getElementById("heroNeed").textContent =
    `Need ${summary.order_gallons ?? "--"} gallons for next ${summary.horizon_days ?? "--"} days`;
  document.getElementById("heroExpected").textContent = `${formatNum(summary.expected_total_gallons, 2)} gal`;
  document.getElementById("heroOrder").textContent = `${summary.order_gallons ?? "--"} gal`;
  document.getElementById("heroModel").textContent = summary.model_used || "--";
  document.getElementById("forecastNote").textContent = summary.note || "";
  const heroInterval = document.getElementById("heroInterval");
  const hasExpectedBand =
    Number.isFinite(toNumber(summary.expected_total_lower_gallons, NaN)) &&
    Number.isFinite(toNumber(summary.expected_total_upper_gallons, NaN)) &&
    Number.isFinite(toNumber(summary.expected_total_pm_gallons, NaN));
  const hasOrderBand =
    Number.isFinite(toNumber(summary.order_lower_gallons, NaN)) &&
    Number.isFinite(toNumber(summary.order_upper_gallons, NaN)) &&
    Number.isFinite(toNumber(summary.order_pm_gallons, NaN));
  if (heroInterval) {
    if (hasExpectedBand && hasOrderBand) {
      heroInterval.textContent =
        `Order range: ${Math.round(toNumber(summary.order_lower_gallons))}-${Math.round(
          toNumber(summary.order_upper_gallons)
        )} gal (+/-${Math.round(toNumber(summary.order_pm_gallons))}); ` +
        `expected usage: ${formatNum(summary.expected_total_lower_gallons, 2)}-${formatNum(
          summary.expected_total_upper_gallons,
          2
        )} gal (+/-${formatNum(summary.expected_total_pm_gallons, 2)}).`;
    } else {
      heroInterval.textContent = "Interval: --";
    }
  }

  const traces = [
    {
      x: past.map((r) => r.date),
      y: past.map((r) => toNumber(r.gallons)),
      mode: "lines+markers",
      name: "Past Usage",
      line: { width: 2.2, color: colors.past },
      marker: { size: 5 },
      hovertemplate: "<b>%{x}</b><br>Past usage: %{y:.2f} gal<extra></extra>",
    },
    {
      x: forecast.map((r) => r.date),
      y: forecast.map((r) => toNumber(r.forecast_gallons)),
      mode: "lines+markers",
      name: "Forecast",
      line: { width: 2.2, color: colors.forecast, dash: "dot" },
      marker: forecastMarker,
      hovertemplate: "<b>%{x}</b><br>Forecast: %{y:.2f} gal<extra></extra>",
    },
  ];

  if (past.length > 0 && forecast.length > 0) {
    traces.push({
      x: [past[past.length - 1].date, forecast[0].date],
      y: [toNumber(past[past.length - 1].gallons), toNumber(forecast[0].forecast_gallons)],
      mode: "lines",
      name: "Transition",
      line: { width: 2, color: colors.forecast, dash: "dot" },
      showlegend: false,
      hoverinfo: "skip",
    });
  }

  const canDrawRange =
    state.settings.showForecastRange &&
    forecast.length > 0 &&
    forecast.every((r) => Number.isFinite(toNumber(r.lower_gallons, NaN)) && Number.isFinite(toNumber(r.upper_gallons, NaN)));

  if (canDrawRange) {
    traces.push({
      x: forecast.map((r) => r.date),
      y: forecast.map((r) => toNumber(r.upper_gallons)),
      mode: "lines",
      line: { width: 0 },
      name: "Upper",
      showlegend: false,
      hoverinfo: "skip",
    });
    traces.push({
      x: forecast.map((r) => r.date),
      y: forecast.map((r) => toNumber(r.lower_gallons)),
      mode: "lines",
      line: { width: 0 },
      fill: "tonexty",
      fillcolor: colors.range,
      name: "Forecast Range",
    });
  }

  const sharedYRange = computeSharedYRange([
    past.map((r) => r.gallons),
    forecast.map((r) => r.forecast_gallons),
    forecast.map((r) => r.lower_gallons),
    forecast.map((r) => r.upper_gallons),
  ]);

  const combinedChart = document.getElementById("combinedForecastChart");
  const splitWrap = document.getElementById("splitForecastWrap");
  if (state.settings.splitForecastCharts) {
    combinedChart.classList.add("hidden");
    splitWrap.classList.remove("hidden");

    const pastLayout = plotLayout("");
    pastLayout.showlegend = false;
    pastLayout.xaxis.tickangle = -28;
    if (sharedYRange) pastLayout.yaxis.range = sharedYRange;

    Plotly.react(
      "pastOnlyChart",
      [
        {
          x: past.map((r) => r.date),
          y: past.map((r) => toNumber(r.gallons)),
          mode: "lines+markers",
          name: "Past Usage",
          line: { width: 2.8, color: colors.past },
          marker: { size: 6 },
          hovertemplate: "<b>%{x}</b><br>Past usage: %{y:.2f} gal<extra></extra>",
        },
      ],
      pastLayout,
      { responsive: true, displayModeBar: false }
    );

    const forecastTraces = [
      {
        x: forecast.map((r) => r.date),
        y: forecast.map((r) => toNumber(r.forecast_gallons)),
        mode: "lines+markers",
        name: "Forecast",
        line: { width: 2.8, color: colors.forecast, dash: "dot" },
        marker: forecastMarker,
        hovertemplate: "<b>%{x}</b><br>Forecast: %{y:.2f} gal<extra></extra>",
      },
    ];
    if (canDrawRange) {
      forecastTraces.push({
        x: forecast.map((r) => r.date),
        y: forecast.map((r) => toNumber(r.upper_gallons)),
        mode: "lines",
        line: { width: 0 },
        name: "Upper",
        showlegend: false,
        hoverinfo: "skip",
      });
      forecastTraces.push({
        x: forecast.map((r) => r.date),
        y: forecast.map((r) => toNumber(r.lower_gallons)),
        mode: "lines",
        line: { width: 0 },
        fill: "tonexty",
        fillcolor: colors.range,
        name: "Forecast Range",
        showlegend: false,
      });
    }

    const forecastLayout = plotLayout("");
    forecastLayout.showlegend = false;
    forecastLayout.xaxis.tickangle = -28;
    if (sharedYRange) forecastLayout.yaxis.range = sharedYRange;

    Plotly.react("forecastOnlyChart", forecastTraces, forecastLayout, {
      responsive: true,
      displayModeBar: false,
    });
  } else {
    combinedChart.classList.remove("hidden");
    splitWrap.classList.add("hidden");
    const layout = plotLayout("");
    layout.xaxis.tickangle = -28;
    if (sharedYRange) layout.yaxis.range = sharedYRange;
    Plotly.react("combinedForecastChart", traces, layout, { responsive: true, displayModeBar: false });
  }

  renderWeekdayPatternChart(weekday);
  renderMonthlyPatternChart(monthly);
  renderForecastTables(past, forecast, weekday, monthly);
  setTimeout(resizeVisibleCharts, 40);
}

function renderWeekdayPatternChart(weekdayPayload) {
  const mode = state.settings.weekdayMode;
  const colors = palette();
  if (mode === "boxplot") {
    const traces = (weekdayPayload.boxplot || []).map((entry, idx) => ({
      y: entry.values || [],
      type: "box",
      name: entry.weekday,
      marker: { color: colors.drillBars[idx % colors.drillBars.length] },
      line: { color: colors.sales },
      boxmean: true,
    }));
    Plotly.react("weekdayPatternChart", traces, plotLayout("Distribution by weekday"), {
      responsive: true,
      displayModeBar: false,
    });
    return;
  }
  const rows = weekdayPayload[mode] || [];
  Plotly.react(
    "weekdayPatternChart",
    [
      {
        x: rows.map((r) => r.weekday),
        y: rows.map((r) => toNumber(r.value)),
        type: "bar",
        marker: { color: colors.weekday },
        name: mode === "median" ? "Median gallons" : "Mean gallons",
        hovertemplate: "<b>%{x}</b><br>%{y:.2f} gallons<extra></extra>",
      },
    ],
    plotLayout(""),
    { responsive: true, displayModeBar: false }
  );
}

function renderMonthlyPatternChart(rows) {
  const colors = palette();
  const layout = plotLayout("");
  layout.yaxis2 = {
    overlaying: "y",
    side: "right",
    showgrid: false,
    title: "Avg/day",
  };
  Plotly.react(
    "monthlyPatternChart",
    [
      {
        x: rows.map((r) => r.month),
        y: rows.map((r) => toNumber(r.total_gallons)),
        type: "bar",
        name: "Total gallons",
        marker: { color: colors.monthBar, opacity: colors.barOpacity },
        hovertemplate: "<b>%{x}</b><br>Total gallons: %{y:.2f}<extra></extra>",
      },
      {
        x: rows.map((r) => r.month),
        y: rows.map((r) => toNumber(r.avg_gallons_per_day)),
        type: "scatter",
        mode: "lines+markers",
        yaxis: "y2",
        name: "Avg/day",
        line: { color: colors.monthLine, width: 3.8 },
        marker: { size: 6 },
        hovertemplate: "<b>%{x}</b><br>Avg/day: %{y:.2f}<extra></extra>",
      },
    ],
    layout,
    { responsive: true, displayModeBar: false }
  );
}

function renderForecastTables(past, forecast, weekday, monthly) {
  const pastWithDay = past.map((row) => ({ ...row, day: weekdayLabel(row.date) }));
  const forecastWithDay = forecast.map((row) => ({ ...row, day: weekdayLabel(row.date) }));

  el.forecastTablesWrap.innerHTML =
    renderTable("Past Window", pastWithDay, [
      { key: "date", label: "Date" },
      { key: "day", label: "Day" },
      { key: "gallons", label: "Gallons" },
    ]) +
    renderTable("Forecast Window", forecastWithDay, [
      { key: "date", label: "Date" },
      { key: "day", label: "Day" },
      { key: "forecast_gallons", label: "Forecast" },
      { key: "lower_gallons", label: "Low" },
      { key: "upper_gallons", label: "High" },
    ]);

  el.patternTablesWrap.innerHTML =
    renderTable("Weekday Mean", weekday.mean || [], [
      { key: "weekday", label: "Weekday" },
      { key: "value", label: "Mean Gallons" },
    ]) +
    renderTable("Weekday Median", weekday.median || [], [
      { key: "weekday", label: "Weekday" },
      { key: "value", label: "Median Gallons" },
    ]) +
    renderTable("Monthly Pattern", monthly, [
      { key: "month", label: "Month" },
      { key: "total_gallons", label: "Total Gallons" },
      { key: "avg_gallons_per_day", label: "Avg/Day" },
      { key: "days", label: "Days" },
    ]);
}

function populateCategorySelect(categories, selectedCategory) {
  const current = selectedCategory || "";
  const categoryOptions = categories.map((c) => `<option value="${c}">${c}</option>`).join("");
  el.salesCategorySelect.innerHTML = `<option value="">No drilldown</option>${categoryOptions}`;
  if (current && categories.includes(current)) {
    el.salesCategorySelect.value = current;
  } else {
    el.salesCategorySelect.value = "";
  }
  state.salesFilters.drillCategory = el.salesCategorySelect.value || "";
}

function renderSales() {
  const payload = state.sales;
  if (!payload) return;
  const colors = palette();

  const summary = payload.summary || {};
  const windowInfo = payload.window || {};
  const categoryDual = payload.category_dual_axis || [];
  const comp = payload.category_composition_over_time || [];
  const drill = payload.item_drilldown_over_time || [];
  const topItems = payload.top_items || [];
  const categories = payload.categories || [];

  document.getElementById("salesRowsMetric").textContent = summary.total_sales_rows ?? "--";
  document.getElementById("salesQtyMetric").textContent = formatNum(summary.total_qty, 0);
  document.getElementById("salesAmountMetric").textContent = formatCurrency(summary.total_sales_amount);
  document.getElementById("salesCoverageMetric").textContent = `${formatNum(summary.mapping_coverage_pct, 2)}%`;

  populateCategorySelect(categories, summary.selected_category);
  setStatus(
    el.salesWindowStatus,
    `Window: ${windowInfo.start_date || "--"} to ${windowInfo.end_date || "--"} (${windowInfo.mode || "all_time"})`
  );

  renderSalesDualAxisChart(categoryDual, colors);
  renderCategoryCompositionChart(comp, colors);
  renderItemDrilldownChart(drill, summary.selected_category, colors);

  el.salesTablesWrap.innerHTML =
    renderTable("Summary", [
      {
        date_min: summary.date_min || "--",
        date_max: summary.date_max || "--",
        total_qty: formatNum(summary.total_qty, 2),
        total_sales_amount: formatCurrency(summary.total_sales_amount),
        coverage: `${formatNum(summary.mapping_coverage_pct, 2)}%`,
      },
    ], [
      { key: "date_min", label: "Date Min" },
      { key: "date_max", label: "Date Max" },
      { key: "total_qty", label: "Qty" },
      { key: "total_sales_amount", label: "Sales $" },
      { key: "coverage", label: "Coverage" },
    ]) +
    renderTable("Category Qty + Sales", categoryDual, [
      { key: "category", label: "Category" },
      { key: "total_qty", label: "Qty" },
      { key: "total_sales_amount", label: "Sales $" },
    ]) +
    renderTable("Top Items in Window", topItems, [
      { key: "category", label: "Category" },
      { key: "item", label: "Item" },
      { key: "total_qty", label: "Qty" },
      { key: "total_sales_amount", label: "Sales $" },
    ]) +
    renderTable("Category Composition Over Time", comp, [
      { key: "month", label: "Month" },
      { key: "category", label: "Category" },
      { key: "total_qty", label: "Qty" },
      { key: "share_pct", label: "Share %" },
    ]) +
    renderTable("Item Drilldown Over Time", drill, [
      { key: "month", label: "Month" },
      { key: "item", label: "Item" },
      { key: "total_qty", label: "Qty" },
      { key: "total_sales_amount", label: "Sales $" },
    ]);
}

function renderSalesDualAxisChart(rows, colors) {
  const dark = document.body.getAttribute("data-theme") === "dark";
  const layout = plotLayout("");
  layout.height = 430;
  layout.xaxis.tickangle = -30;
  layout.margin.t = 72;
  layout.margin.b = 112;
  layout.margin.r = 62;
  layout.legend = {
    orientation: "h",
    x: 0,
    xanchor: "left",
    y: 1.08,
    yanchor: "bottom",
    bgcolor: "rgba(0,0,0,0)",
    font: { size: 11 },
  };
  layout.yaxis.title = "Volume (Qty)";
  layout.yaxis2 = {
    title: "Sales Amount ($)",
    overlaying: "y",
    side: "right",
    showgrid: false,
    automargin: true,
  };
  Plotly.react(
    "categoryDualAxisChart",
    [
      {
        x: rows.map((r) => r.category),
        y: rows.map((r) => toNumber(r.total_qty)),
        type: "bar",
        name: "Sales Volume",
        marker: {
          color: rows.map((_, idx) => colors.salesBars[idx % colors.salesBars.length]),
          opacity: colors.barOpacity,
          line: { color: dark ? "rgba(248,250,252,0.38)" : "rgba(15,23,42,0.28)", width: 1.1 },
        },
        hovertemplate: "<b>%{x}</b><br>Sales Volume: %{y:.2f}<extra></extra>",
      },
      {
        x: rows.map((r) => r.category),
        y: rows.map((r) => toNumber(r.total_sales_amount)),
        type: "scatter",
        mode: "lines+markers",
        yaxis: "y2",
        name: "Sales $",
        line: { color: colors.salesLine, width: 3.2 },
        marker: { size: 7, color: colors.salesLine },
        hovertemplate: "<b>%{x}</b><br>Sales $: %{y:$,.2f}<extra></extra>",
      },
    ],
    layout,
    { responsive: true, displayModeBar: false }
  );
}

function renderCategoryCompositionChart(rows, colors) {
  const dark = document.body.getAttribute("data-theme") === "dark";
  const months = [...new Set(rows.map((r) => r.month))].sort();
  const categories = [...new Set(rows.map((r) => r.category))];
  if (!months.length || !categories.length) {
    Plotly.react("categoryCompositionChart", [], plotLayout(""), { responsive: true, displayModeBar: false });
    return;
  }
  const traces = categories.map((category, idx) => {
    const categoryRows = rows.filter((r) => r.category === category);
    const byMonth = new Map(categoryRows.map((r) => [r.month, toNumber(r.share_pct)]));
    return {
      x: months,
      y: months.map((m) => byMonth.get(m) || 0),
      stackgroup: "one",
      mode: "lines",
      line: { width: 1.8, color: colors.comp[idx % colors.comp.length] },
      fillcolor: hexToRgba(colors.comp[idx % colors.comp.length], dark ? 0.62 : 0.52),
      name: category,
      hovertemplate: "%{x}<br>%{fullData.name}: %{y:.2f}%<extra></extra>",
    };
  });
  const layout = plotLayout("");
  layout.height = 430;
  layout.yaxis.title = "Share %";
  layout.yaxis.range = [0, 100];
  layout.xaxis.tickangle = -20;
  layout.legend = {
    orientation: "h",
    x: 0,
    xanchor: "left",
    y: 1.08,
    yanchor: "bottom",
    bgcolor: "rgba(0,0,0,0)",
    font: { size: 11 },
  };
  layout.margin.t = 78;
  layout.margin.r = 46;
  layout.margin.b = 98;
  Plotly.react("categoryCompositionChart", traces, layout, { responsive: true, displayModeBar: false });
}

function renderItemDrilldownChart(rows, selectedCategory, colors) {
  const dark = document.body.getAttribute("data-theme") === "dark";
  const layout = plotLayout("");
  if (!selectedCategory || !rows.length) {
    layout.hovermode = "closest";
    layout.annotations = [
      {
        text: "Select a category in the sidebar to enable item drilldown",
        showarrow: false,
        x: 0.5,
        y: 0.5,
        xref: "paper",
        yref: "paper",
        font: { size: 14, color: getComputedStyle(document.body).getPropertyValue("--muted").trim() },
      },
    ];
    layout.xaxis = { visible: false };
    layout.yaxis = { visible: false };
    Plotly.react("itemDrilldownChart", [], layout, { responsive: true, displayModeBar: false });
    return;
  }

  const months = [...new Set(rows.map((r) => r.month))].sort();
  const items = [...new Set(rows.map((r) => r.item))];
  const barTraces = items.map((item, idx) => {
    const itemRows = rows.filter((r) => r.item === item);
    const byMonth = new Map(itemRows.map((r) => [r.month, toNumber(r.total_qty)]));
    return {
      x: months,
      y: months.map((m) => byMonth.get(m) || 0),
      type: "bar",
      name: item,
      marker: {
        color: colors.drillBars[idx % colors.drillBars.length],
        opacity: colors.barOpacity,
        line: { color: dark ? "rgba(248,250,252,0.32)" : "rgba(15,23,42,0.24)", width: 1 },
      },
      hovertemplate: "<b>%{x}</b><br>%{fullData.name}: %{y:.2f}<extra></extra>",
    };
  });
  const monthSales = months.map((month) =>
    rows
      .filter((r) => r.month === month)
      .reduce((sum, row) => sum + toNumber(row.total_sales_amount), 0)
  );
  const salesTrace = {
    x: months,
    y: monthSales,
    type: "scatter",
    mode: "lines+markers",
    yaxis: "y2",
    name: "Sales $ Total",
    line: { color: colors.drillLine, width: 3.3 },
    marker: { size: 7, color: colors.drillLine },
    hovertemplate: "<b>%{x}</b><br>Sales $ Total: %{y:$,.2f}<extra></extra>",
  };

  layout.barmode = "stack";
  layout.hovermode = "closest";
  layout.height = 430;
  layout.xaxis.tickangle = -20;
  layout.margin.t = 82;
  layout.yaxis.title = "Volume (Qty)";
  layout.yaxis2 = {
    title: "Sales Amount ($)",
    overlaying: "y",
    side: "right",
    showgrid: false,
    automargin: true,
  };
  layout.legend = {
    orientation: "h",
    x: 0,
    xanchor: "left",
    y: 1.08,
    yanchor: "bottom",
    bgcolor: "rgba(0,0,0,0)",
    font: { size: 11 },
  };
  layout.margin.r = 58;
  layout.margin.b = 110;
  Plotly.react("itemDrilldownChart", [...barTraces, salesTrace], layout, {
    responsive: true,
    displayModeBar: false,
  });
}

function ingredientRowTemplate(row = {}) {
  return `
    <tr>
      <td><input type="text" value="${(row.category || "").replace(/"/g, "&quot;")}" /></td>
      <td><input type="text" value="${(row.item || "").replace(/"/g, "&quot;")}" /></td>
      <td><input type="text" value="${(row.price_point_name || "").replace(/"/g, "&quot;")}" /></td>
      <td><input type="number" step="0.1" value="${toNumber(row.milk_oz, 0)}" /></td>
      <td><button class="delete-row-btn" type="button">Delete</button></td>
    </tr>
  `;
}

function renderIngredientsTable() {
  const rows = state.ingredients || [];
  el.ingredientTableBody.innerHTML = rows.map((r) => ingredientRowTemplate(r)).join("");
  resetIngredientSearchState();
}

function gatherIngredientRows() {
  const rows = [];
  el.ingredientTableBody.querySelectorAll("tr").forEach((tr) => {
    const inputs = tr.querySelectorAll("input");
    rows.push({
      category: inputs[0].value.trim(),
      item: inputs[1].value.trim(),
      price_point_name: inputs[2].value.trim(),
      milk_oz: toNumber(inputs[3].value, 0),
    });
  });
  return rows;
}

function bindTabNavigation() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab)?.classList.add("active");
      setTimeout(resizeVisibleCharts, 60);
    });
  });
}

function bindIngredientsTableActions() {
  el.ingredientTableBody.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.classList.contains("delete-row-btn")) {
      target.closest("tr")?.remove();
      if (state.ingredientSearch.query) {
        runIngredientSearch();
      }
    }
  });
}

async function runFullRefresh() {
  readSettingsFromInputs();
  readSalesFiltersFromInputs();
  setStatus(el.sidebarStatus, "Loading dashboard data...");
  try {
    await Promise.all([loadForecast(), loadSales(), loadIngredients()]);
    setStatus(el.sidebarStatus, "Dashboard updated", "success");
  } catch (error) {
    setStatus(el.sidebarStatus, error.message || "Failed to load dashboard", "error");
  }
}

function bindEvents() {
  [el.horizonDays, el.lookbackDays, el.safetyBufferPct, el.topNItems].forEach((input) => {
    input.addEventListener("input", readSettingsFromInputs);
  });

  el.forecastModel.addEventListener("change", () => {
    readSettingsFromInputs();
  });

  el.showForecastRange.addEventListener("change", () => {
    readSettingsFromInputs();
    if (state.forecast) renderForecast();
  });

  el.splitForecastCharts.addEventListener("change", () => {
    readSettingsFromInputs();
    if (state.forecast) renderForecast();
  });

  el.weekdayMode.addEventListener("change", () => {
    readSettingsFromInputs();
    if (state.forecast) renderForecast();
  });

  el.themeSelect.addEventListener("change", () => applyTheme(el.themeSelect.value));

  el.sidebarToggleBtn.addEventListener("click", () => {
    el.appShell.classList.toggle("sidebar-collapsed");
    setTimeout(resizeVisibleCharts, 180);
  });

  el.applySettingsBtn.addEventListener("click", async () => {
    await runFullRefresh();
  });

  el.refreshDataBtn.addEventListener("click", async () => {
    setStatus(el.sidebarStatus, "Refreshing server cache...");
    try {
      await fetchJSON("/api/dashboard/refresh", { method: "POST" });
      await runFullRefresh();
    } catch (error) {
      setStatus(el.sidebarStatus, error.message || "Refresh failed", "error");
    }
  });

  if (el.authRegisterBtn) {
    el.authRegisterBtn.addEventListener("click", () => {
      openRegisterModal();
    });
  }

  if (el.registerSubmitBtn) {
    el.registerSubmitBtn.addEventListener("click", async () => {
      try {
        await registerAuth();
      } catch (error) {
        setStatus(el.registerStatus, error.message || "Register failed", "error");
      }
    });
  }

  [el.registerModalCloseBtn, el.registerCancelBtn].forEach((btn) => {
    if (btn) {
      btn.addEventListener("click", closeRegisterModal);
    }
  });

  if (el.registerModal) {
    el.registerModal.addEventListener("click", (event) => {
      if (event.target === el.registerModal) {
        closeRegisterModal();
      }
    });
  }

  document.addEventListener("keydown", (event) => {
    const registerOpen = el.registerModal && !el.registerModal.classList.contains("hidden");
    if (!registerOpen) return;
    if (event.key === "Escape") {
      closeRegisterModal();
    }
    if (event.key === "Enter") {
      const target = event.target;
      if (target instanceof HTMLInputElement && el.registerModal.contains(target)) {
        event.preventDefault();
        el.registerSubmitBtn?.click();
      }
    }
  });

  [
    [el.authEmail, el.authEmailError],
    [el.authPassword, el.authPasswordError],
    [el.registerFirstName, el.registerFirstNameError],
    [el.registerLastName, el.registerLastNameError],
    [el.registerEmail, el.registerEmailError],
    [el.registerPassword, el.registerPasswordError],
  ].forEach(([input, errorNode]) => {
    if (input) {
      input.addEventListener("input", () => setFieldError(input, errorNode));
    }
  });

  if (el.authLoginBtn) {
    el.authLoginBtn.addEventListener("click", async () => {
      try {
        await loginAuth();
      } catch (error) {
        setStatus(el.authStatus, error.message || "Login failed", "error");
      }
    });
  }

  if (el.authLogoutBtn) {
    el.authLogoutBtn.addEventListener("click", async () => {
      try {
        await logoutAuth();
      } catch (error) {
        setStatus(el.authStatus, error.message || "Logout failed", "error");
      }
    });
  }

  if (el.squareConnectBtn) {
    el.squareConnectBtn.addEventListener("click", () => {
      window.location.href = "/api/auth/square/connect";
    });
  }

  if (el.squareDisconnectBtn) {
    el.squareDisconnectBtn.addEventListener("click", async () => {
      try {
        await fetchJSON("/api/auth/square/disconnect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ revoke_in_square: true }),
        });
        await loadAuthMe();
      } catch (error) {
        setStatus(el.authStatus, error.message || "Disconnect failed", "error");
      }
    });
  }

  if (el.adminLoadConfigBtn) {
    el.adminLoadConfigBtn.addEventListener("click", async () => {
      setStatus(el.adminModelStatus, "Loading model config...");
      try {
        await loadAdminModelConfig();
      } catch (error) {
        setStatus(el.adminModelStatus, error.message || "Failed to load model config", "error");
      }
    });
  }

  if (el.adminSaveConfigBtn) {
    el.adminSaveConfigBtn.addEventListener("click", async () => {
      setStatus(el.adminModelStatus, "Saving model config...");
      el.adminSaveConfigBtn.disabled = true;
      try {
        await saveAdminModelConfig();
        await runFullRefresh();
      } catch (error) {
        setStatus(el.adminModelStatus, error.message || "Failed to save model config", "error");
      } finally {
        el.adminSaveConfigBtn.disabled = false;
      }
    });
  }

  if (el.adminRunEodBtn) {
    el.adminRunEodBtn.addEventListener("click", async () => {
      setStatus(el.adminEodStatus, "Running EOD sync...");
      el.adminRunEodBtn.disabled = true;
      try {
        await runEodSync();
      } catch (error) {
        setStatus(el.adminEodStatus, error.message || "EOD sync failed", "error");
      } finally {
        el.adminRunEodBtn.disabled = false;
      }
    });
  }

  if (el.adminLoadLatestEodBtn) {
    el.adminLoadLatestEodBtn.addEventListener("click", async () => {
      setStatus(el.adminEodStatus, "Loading latest EOD sync...");
      try {
        await loadLatestEodSync();
      } catch (error) {
        setStatus(el.adminEodStatus, error.message || "Failed to load latest sync", "error");
      }
    });
  }

  el.salesWindowMode.addEventListener("change", () => {
    toggleCustomDateInputs();
    readSalesFiltersFromInputs();
  });

  el.applySalesWindowBtn.addEventListener("click", async () => {
    setStatus(el.salesWindowStatus, "Applying sales window...");
    try {
      await loadSales();
      setStatus(el.salesWindowStatus, "Sales window applied", "success");
    } catch (error) {
      setStatus(el.salesWindowStatus, error.message || "Could not apply sales window", "error");
    }
  });

  el.salesCategorySelect.addEventListener("change", async () => {
    readSalesFiltersFromInputs();
    try {
      await loadSales();
    } catch (error) {
      setStatus(el.salesWindowStatus, error.message || "Could not refresh drilldown", "error");
    }
  });

  el.addIngredientRowBtn.addEventListener("click", () => {
    el.ingredientTableBody.insertAdjacentHTML("beforeend", ingredientRowTemplate());
    if (state.ingredientSearch.query) {
      runIngredientSearch();
    }
  });

  if (el.ingredientSearchBtn) {
    el.ingredientSearchBtn.addEventListener("click", () => {
      runIngredientSearch();
    });
  }

  if (el.ingredientSearch) {
    el.ingredientSearch.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        runIngredientSearch();
      }
    });
  }

  el.reloadIngredientBtn.addEventListener("click", async () => {
    setStatus(el.ingredientStatus, "Reloading ingredient rows...");
    try {
      await loadIngredients();
      setStatus(el.ingredientStatus, "Ingredient table reloaded", "success");
    } catch (error) {
      setStatus(el.ingredientStatus, error.message || "Reload failed", "error");
    }
  });

  el.saveIngredientBtn.addEventListener("click", async () => {
    setStatus(el.ingredientStatus, "Saving ingredient mappings...");
    el.saveIngredientBtn.disabled = true;
    try {
      const payload = await fetchJSON("/api/dashboard/ingredients", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows: gatherIngredientRows() }),
      });
      setStatus(el.ingredientStatus, `Saved ${payload.rows_written} rows to ${payload.file_name}`, "success");
      await runFullRefresh();
    } catch (error) {
      setStatus(el.ingredientStatus, error.message || "Save failed", "error");
    } finally {
      el.saveIngredientBtn.disabled = false;
    }
  });

  window.addEventListener("resize", () => {
    resizeVisibleCharts();
  });
}

async function init() {
  bindTabNavigation();
  bindIngredientsTableActions();
  bindEvents();
  readSettingsFromInputs();
  readSalesFiltersFromInputs();
  toggleCustomDateInputs();
  applyTheme(el.themeSelect.value);
  try {
    await loadAuthMe();
  } catch (error) {
    setStatus(el.authStatus, error.message || "Could not load auth state", "error");
  }
  await runFullRefresh();
  try {
    await loadAdminModelConfig();
  } catch (error) {
    setStatus(el.adminModelStatus, error.message || "Admin config unavailable");
  }
  try {
    await loadLatestEodSync();
  } catch (error) {
    setStatus(el.adminEodStatus, error.message || "No latest EOD sync");
  }
}

init();
