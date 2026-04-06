const API_BASE = "http://127.0.0.1:8000";

const State = {
    buyer_id: null,
    session_id: null,
    catalog: [],              // raw catalog products from /catalog
    costMap: {},              // product_id → cost_price (for price calculation)
    pricing: {},              // product_id → current AI MULTIPLIER (e.g. 1.5)
    prev_pricing: {},         // product_id → previous AI MULTIPLIER (for trend arrows)
    step: 0,
    sessionDone: false
};

// --- Product Icon Mapping ---
const iconMap = {
    'milk':         'fa-bottle-water text-indigo',
    'bread':        'fa-bread-slice text-warning',
    'chips':        'fa-cookie-bite text-success',
    'strawberries': 'fa-seedling text-pink'
};

// --- AI Reasoning Map (using multiplier delta) ---
function getAIReason(productId, newMult, oldMult) {
    if (oldMult === undefined) {
        return '<i class="fa-solid fa-scale-balanced"></i> <strong>Session Initialised</strong> – AI set opening prices';
    }
    const diff = newMult - oldMult;
    if (diff > 0.08) {
        return '<i class="fa-solid fa-arrow-trend-up text-indigo"></i> <strong>Raised Price</strong> – Demand surging / stock healthy';
    }
    if (diff < -0.08) {
        return '<i class="fa-solid fa-arrow-trend-down text-warning"></i> <strong>Slashed Price</strong> – Nearing expiry / clearing stock';
    }
    return '<i class="fa-solid fa-minus" style="color:var(--text-muted)"></i> <strong>Holding</strong> – Optimal market equilibrium';
}

// --- Initialization ---
document.addEventListener("DOMContentLoaded", () => {
    fetchCatalog();
});

// --- UI Helpers ---
function showToast(message, type = "success") {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// --- Fetch Catalog (seeds costMap and initial pricing) ---
async function fetchCatalog() {
    try {
        const res = await fetch(`${API_BASE}/catalog`);
        const data = await res.json();
        State.catalog = data.products || [];

        // Build cost map for price calculation later
        State.catalog.forEach(p => {
            State.costMap[p.product_id] = p.cost_price;
            // Default markup 1.3x until AI acts
            State.pricing[p.product_id] = 1.3;
        });

        renderCatalog();
    } catch (e) {
        showToast("Error fetching catalog. Is the server running?", "error");
    }
}

// --- 1. Buyer Registration ---
document.getElementById('form-register-buyer').addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
        name:    document.getElementById('buyer-name').value,
        email:   document.getElementById('buyer-email').value,
        address: document.getElementById('buyer-address').value
    };
    try {
        const res = await fetch(`${API_BASE}/buyers/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.buyer_id) {
            State.buyer_id = data.buyer_id;
            const statusEl = document.getElementById('buyer-status');
            statusEl.textContent = `✓ Active: ${data.buyer_id.substring(0, 8)}`;
            statusEl.className = 'setup-state success';
            showToast(`Welcome, ${payload.name}!`);
            renderCatalog();
        }
    } catch (e) {
        showToast("Failed to register player", "error");
    }
});

// --- 2. Boot Environment ---
document.getElementById('btn-start-session').addEventListener('click', async () => {
    const taskName = document.getElementById('rl-task').value;
    const btn = document.getElementById('btn-start-session');
    btn.disabled = true;
    btn.textContent = '⏳ Booting...';

    try {
        const res = await fetch(`${API_BASE}/session/new`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'x-api-key': 'dev-client-key' },
            body: JSON.stringify({ task_name: taskName })
        });
        const data = await res.json();

        if (data.session_id) {
            State.session_id  = data.session_id;
            State.step        = 0;
            State.sessionDone = false;

            // Reset pricing to default 1.3x so product cards don't carry stale AI values
            State.prev_pricing = {};
            State.catalog.forEach(p => { State.pricing[p.product_id] = 1.3; });

            document.getElementById('metric-steps').textContent = '0';
            document.getElementById('metric-score').textContent = '—';
            document.getElementById('ai-log').textContent       = 'Environment ready. Click "Advance" to let the AI act.';
            document.getElementById('financial-breakdown').innerHTML = '<em>Financial breakdown will appear after first turn.</em>';

            const statusEl = document.getElementById('session-status');
            statusEl.textContent  = `▶ ${taskName}`;
            statusEl.className    = 'setup-state success';

            document.getElementById('btn-rl-decide').disabled = false;

            // Also re-render the catalog so it shows clean default prices
            renderCatalog();
            showToast(`Environment Booted: ${taskName}`);
        } else {
            showToast(`Boot failed: ${data.detail || 'unknown error'}`, 'error');
        }
    } catch (e) {
        showToast("Failed to boot environment", "error");
    } finally {
        btn.disabled    = false;
        btn.textContent = 'Boot Environment';
    }
});

// --- 3. Place Order (Human Buyer) ---
window.placeOrder = async function(productId) {
    if (!State.buyer_id) {
        showToast("⚠️ Please Sign In as a Buyer first (sidebar).", "warning");
        return;
    }

    const btn = document.getElementById(`btn-buy-${productId}`);
    btn.disabled   = true;
    btn.innerHTML  = '<i class="fa-solid fa-spinner fa-spin"></i>';

    const payload = {
        buyer_id:         State.buyer_id,
        product_id:       productId,
        quantity:         1,
        delivery_address: document.getElementById('buyer-address').value
    };

    // If a session is active, pass session_id so the order deducts from RL inventory
    const url = State.session_id
        ? `${API_BASE}/orders?session_id=${State.session_id}`
        : `${API_BASE}/orders`;

    try {
        const res = await fetch(url, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(payload)
        });
        if (res.ok) {
            const mult  = State.pricing[productId] || 1.3;
            const cost  = State.costMap[productId]  || 0;
            const price = (cost * mult).toFixed(2);
            showToast(`🛒 Ordered 1× ${productId} at $${price}`);
        } else {
            const err = await res.json();
            showToast(err.detail || "Failed to place order.", "error");
        }
    } catch (e) {
        showToast("Error communicating with API", "error");
    } finally {
        btn.disabled   = false;
        btn.innerHTML  = 'Order 1 Unit';
    }
};

// --- 4. Advance AI Turn ---
document.getElementById('btn-rl-decide').addEventListener('click', async () => {
    if (!State.session_id || State.sessionDone) return;

    const btn      = document.getElementById('btn-rl-decide');
    btn.disabled   = true;
    btn.innerHTML  = '<i class="fa-solid fa-microchip fa-spin"></i> Simulating...';

    const payload = {
        session_id:     State.session_id,
        use_curriculum: document.getElementById('RL-curriculum').checked,
        stochastic:     document.getElementById('RL-stochastic').checked
    };

    try {
        const res  = await fetch(`${API_BASE}/session/rl-decide`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json', 'x-api-key': 'dev-client-key' },
            body:    JSON.stringify(payload)
        });
        const data = await res.json();

        if (!res.ok) {
            showToast(data.detail || 'API error on advance', 'error');
            return;
        }

        State.step++;
        document.getElementById('metric-steps').textContent = State.step;

        // --- Update multipliers (AI pricing output IS a multiplier, not a dollar amount) ---
        if (data.action && data.action.pricing) {
            State.prev_pricing = { ...State.pricing };
            // The API returns multipliers directly (e.g. milk: 3.0 means 3.0x cost)
            Object.assign(State.pricing, data.action.pricing);
            renderCatalog();
        }

        // --- Show per-turn profit as dollar amount ---
        if (data.reward !== undefined) {
            const sign = data.reward >= 0 ? '+' : '';
            document.getElementById('metric-score').textContent = `${sign}$${data.reward.toFixed(2)}`;
        }

        // --- Financial Breakdown ---
        if (data.reward_breakdown) {
            const b       = data.reward_breakdown;
            const sales   = b.successful_sale_reward.toFixed(2);
            const bonus   = b.efficiency_bonus.toFixed(2);
            const overhead= b.overhead_penalty.toFixed(2);
            const waste   = b.waste_penalty.toFixed(2);
            const net     = data.reward.toFixed(2);
            const netSign = data.reward >= 0 ? '+' : '';

            document.getElementById('financial-breakdown').innerHTML =
                `<strong style="color:#fff;">Turn Breakdown:</strong>
                 <span style="color:var(--success)">+$${sales} Sales</span>
                 ${parseFloat(bonus) > 0 ? `<span style="color:var(--success)"> +$${bonus} Bonus</span>` : ''}
                 | <span style="color:var(--error)">-$${overhead} Overhead</span>
                 | <span style="color:var(--error)">-$${waste} Spoilage</span>
                 = <strong style="color:#fff;">${netSign}$${net}</strong>`;
        }

        // --- AI decision log: show REAL prices, not raw multipliers ---
        const priceStrings = Object.entries(data.action.pricing).map(([pid, mult]) => {
            const cost  = State.costMap[pid] || 0;
            const price = (cost * parseFloat(mult)).toFixed(2);
            return `${pid}: $${price}`;
        }).join(' | ');
        document.getElementById('ai-log').innerHTML =
            `Step ${State.step}: <span style="color:var(--brand-accent)">${priceStrings}</span>`;

        // --- Episode done? ---
        if (data.done) {
            State.sessionDone = true;
            btn.disabled      = true;
            btn.innerHTML     = '<i class="fa-solid fa-flag-checkered"></i> Episode Complete';
            showToast(`Episode finished! Final score: ${data.score?.toFixed(3) ?? 'N/A'}`, 'success');
            document.getElementById('metric-score').textContent = `Score: ${data.score?.toFixed(3) ?? '—'}`;
        }

    } catch (e) {
        showToast("AI Execution Failed", "error");
        console.error(e);
    } finally {
        if (!State.sessionDone) {
            btn.disabled  = false;
            btn.innerHTML = '<i class="fa-solid fa-forward-step"></i> Advance 15 mins (AI Turn)';
        }
    }
});

// --- Render Catalog ---
function renderCatalog() {
    const grid = document.getElementById('catalog-grid');
    grid.innerHTML = '';

    if (State.catalog.length === 0) {
        grid.innerHTML = '<div class="glass-panel skeleton-loader" style="padding:2rem;text-align:center;">Loading catalog…</div>';
        return;
    }

    State.catalog.forEach(p => {
        const mult      = State.pricing[p.product_id] ?? 1.3;
        const prevMult  = State.prev_pricing[p.product_id];
        const truePrice = p.cost_price * mult;
        const iconClass = iconMap[p.product_id] || 'fa-box';
        const reason    = getAIReason(p.product_id, mult, prevMult);
        const isLocked  = !State.session_id; // grey out if no session

        const card = document.createElement('div');
        card.className = 'product-card';
        card.innerHTML = `
            <i class="fa-solid ${iconClass} product-icon"></i>
            <h3 class="product-name">${p.name}</h3>
            <p class="product-meta">${p.category.toUpperCase()} • Base Cost $${p.cost_price.toFixed(2)}</p>
            <div class="product-price">$${truePrice.toFixed(2)}</div>
            <p class="product-meta" style="margin-bottom:0.35rem;">Markup: ${(mult * 100).toFixed(0)}%</p>
            <div class="product-meta" style="background:rgba(255,255,255,0.05);padding:0.5rem 0.75rem;border-radius:6px;margin-bottom:1rem;color:#fff;font-size:0.8rem;text-align:left;">
                ${reason}
            </div>
            <button
                id="btn-buy-${p.product_id}"
                class="btn ${isLocked ? 'secondary' : 'primary'} btn-sm"
                onclick="placeOrder('${p.product_id}')"
                ${isLocked ? 'style="opacity:0.5"' : ''}>
                Order 1 Unit
            </button>
        `;
        grid.appendChild(card);
    });
}
