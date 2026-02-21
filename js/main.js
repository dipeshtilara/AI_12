// Main App Logic
// import { curriculumData } from './data/curriculum.js'; // Removed for file:// compat
// import { setupVisualizer } from './components/visualizer.js'; // Removed for file:// compat

class App {
    constructor() {
        this.sidebarNav = document.getElementById('sidebar-nav');
        this.contentArea = document.getElementById('content-area');
        this.breadcrumbs = document.getElementById('breadcrumbs');
        this.menuToggle = document.getElementById('menu-toggle');
        this.sidebar = document.querySelector('.sidebar');

        this.currentUnitIndex = -1;

        this.init();
    }

    init() {
        this.renderSidebar();
        this.createModal(); // Initialize Modal HTML
        this.setupEventListeners();

        // Expose app for inline handlers if needed
        window.app = this;
    }

    createModal() {
        // Inject modal HTML into body
        const modalHTML = `
            <div class="modal-overlay" id="content-modal">
                <div class="modal-content">
                    <div class="modal-header">
                        <div class="modal-title" id="modal-title">Topic Title</div>
                        <button class="modal-close" onclick="app.closeModal()">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                    </div>
                    <div class="modal-body" id="modal-body">
                        <!-- Content goes here -->
                    </div>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', modalHTML);

        this.modalOverlay = document.getElementById('content-modal');
        this.modalTitle = document.getElementById('modal-title');
        this.modalBody = document.getElementById('modal-body');

        // Close on clicking outside
        this.modalOverlay.addEventListener('click', (e) => {
            if (e.target === this.modalOverlay) {
                this.closeModal();
            }
        });
    }

    openModal(title, content) {
        this.modalTitle.textContent = title;
        this.modalBody.innerHTML = content;
        this.modalOverlay.classList.add('open');
    }

    closeModal() {
        this.modalOverlay.classList.remove('open');
    }

    setupEventListeners() {
        this.menuToggle.addEventListener('click', () => {
            this.sidebar.classList.toggle('open');
        });

        // Close sidebar on item click on mobile
        this.sidebarNav.addEventListener('click', (e) => {
            if (window.innerWidth <= 768 && e.target.closest('.nav-item')) {
                this.sidebar.classList.remove('open');
            }
        });

        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.modalOverlay.classList.contains('open')) {
                this.closeModal();
            }
        });
    }

    renderSidebar() {
        this.sidebarNav.innerHTML = '';
        window.curriculumData.forEach((unit, index) => {
            const item = document.createElement('div');
            item.className = 'nav-item';
            item.innerHTML = `
                <div class="nav-icon" style="width: 8px; height: 8px; border-radius: 50%; background-color: ${unit.color}"></div>
                <span>${unit.title.split(':')[0]}</span>
            `;
            item.addEventListener('click', () => this.navigateToUnit(index));
            this.sidebarNav.appendChild(item);
        });
    }

    navigateToUnit(index) {
        this.currentUnitIndex = index;
        const unit = window.curriculumData[index];

        // Update Sidebar Active State
        const items = this.sidebarNav.querySelectorAll('.nav-item');
        items.forEach(i => i.classList.remove('active'));
        items[index].classList.add('active');

        // Update Breadcrumbs
        this.breadcrumbs.textContent = unit.title;

        this.renderUnitContent(unit);
    }

    renderUnitContent(unit) {
        this.contentArea.style.opacity = '0';

        setTimeout(() => {
            // Helper to generate clickable lists
            // Now supports both object structures (new) and string (fallback/old)
            const generateList = (items) => {
                return items.map((item, idx) => {
                    const title = item.title || item; // Handle object or string
                    // Store content in a data attribute? No, too large.
                    // We will bind click event via index.
                    return `<li class="clickable-item" data-type="lo" data-idx="${idx}">${title}</li>`;
                }).join('');
            };

            const generateActivityList = (items) => {
                return items.map((item, idx) => {
                    const title = item.title || item;
                    return `<li class="clickable-item" data-type="act" data-idx="${idx}">${title}</li>`;
                }).join('');
            };

            this.contentArea.innerHTML = `
                <section class="unit-hero" style="border-left: 4px solid ${unit.color}">
                    <h1>${unit.title}</h1>
                    <p>${unit.description}</p>
                </section>

                <div class="visualization-container" id="vis-${unit.id}">
                    <div class="vis-placeholder">
                        <div class="loader">Loading Visualization...</div>
                    </div>
                </div>

                <div class="content-grid">
                    <div class="card learning-outcomes">
                        <h3><span class="icon">🎯</span> Learning Outcomes</h3>
                        <p class="hint-text"><small>(Click topics to view notes)</small></p>
                        <ul id="lo-list">
                            ${generateList(unit.learningOutcomes)}
                        </ul>
                    </div>
                    
                    <div class="card activities">
                        <h3><span class="icon">⚡</span> Activities & Practicals</h3>
                         <p class="hint-text"><small>(Click activities to view details)</small></p>
                        <ul id="act-list">
                            ${generateActivityList(unit.activities)}
                        </ul>
                    </div>
                </div>
            `;

            this.contentArea.style.opacity = '1';

            // Bind Click Events for Modal
            // We do this here because we need reference to the current 'unit' object
            const loList = document.getElementById('lo-list');
            const actList = document.getElementById('act-list');

            loList.addEventListener('click', (e) => {
                const target = e.target.closest('.clickable-item');
                if (target) {
                    const idx = target.dataset.idx;
                    const item = unit.learningOutcomes[idx];
                    this.openModal(item.title || item, item.content || `<p>No detailed content available yet for: ${item}</p>`);
                }
            });

            actList.addEventListener('click', (e) => {
                const target = e.target.closest('.clickable-item');
                if (target) {
                    const idx = target.dataset.idx;
                    const item = unit.activities[idx];
                    this.openModal(item.title || item, item.content || `<p>Activity details coming soon: ${item}</p>`);
                }
            });


            // Initialize Visualization
            if (window.setupVisualizer) {
                window.setupVisualizer(unit.id, `vis-${unit.id}`, unit.color);
            }

        }, 200);
    }
}

// Global Styles for dynamic content styling
const dynamicStyles = document.createElement('style');
dynamicStyles.textContent = `
    .unit-hero {
        padding: 2rem;
        background: linear-gradient(90deg, rgba(255,255,255,0.03), transparent);
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .unit-hero h1 {
        font-family: var(--font-heading);
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        color: var(--text-main);
    }
    .unit-hero p {
        color: var(--text-muted);
        font-size: 1.1rem;
    }
    
    .visualization-container {
        width: 100%;
        height: 400px;
        background: rgba(0,0,0,0.3);
        border-radius: 16px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.05);
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .content-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
    }
    
    .card {
        background: var(--bg-card);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .card h3 {
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--primary);
    }

    .hint-text {
        color: var(--text-muted);
        margin-bottom: 1rem;
        font-style: italic;
        opacity: 0.7;
    }
    
    .card ul {
        list-style: none;
    }
    
    .card li {
        margin-bottom: 0.8rem;
        padding-left: 1.2rem;
        position: relative;
        line-height: 1.5;
        color: var(--text-muted);
    }
    
    .card li::before {
        content: "•";
        color: var(--secondary);
        position: absolute;
        left: 0;
        font-weight: bold;
    }

    @media (max-width: 900px) {
        .content-grid {
            grid-template-columns: 1fr;
        }
    }
`;
document.head.appendChild(dynamicStyles);

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    new App();
});
