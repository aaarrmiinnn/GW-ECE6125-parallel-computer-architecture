# GWU ECE 6125: Parallel Computer Architecture Lecture Series

This repository contains lecture materials for the GWU ECE 6125 Parallel Computer Architecture course. The materials are presented as interactive web slides built with Reveal.js.

## Contents

- [Cache Coherence in Parallel Computer Architecture](#cache-coherence) (Lecture 5)
- More lectures coming soon...

## Features

- Interactive slides with Reveal.js
- Dark/light mode toggle
- Mobile-friendly and responsive design
- PDF download option for offline viewing
- Automatic deployment to GitHub Pages

## Viewing the Lectures

### Online (GitHub Pages)

The lectures are available online at: 
```
https://[YOUR_USERNAME].github.io/[YOUR_REPO_NAME]/
```

### Locally

To run the lecture series locally:

1. Clone this repository:
   ```bash
   git clone https://github.com/[YOUR_USERNAME]/[YOUR_REPO_NAME].git
   cd [YOUR_REPO_NAME]
   ```

2. Start the local server:
   ```bash
   ./start.sh
   ```
   
   This will start a server at http://localhost:8000

3. For a specific port (e.g. 8080):
   ```bash
   ./start.sh 8080
   ```

## Downloading Slides as PDF

Each lecture presentation includes a "Download PDF" button in the top right corner. To save the slides as a PDF:

1. Click the "Download PDF" button
2. When the print dialog opens, select "Save as PDF" as the destination
3. Important print settings:
   - Orientation: **Landscape**
   - Margins: **None** or **Minimum**
   - Scale: **100%**
   - **Check "Background graphics"** to include diagrams and colors
   - **Uncheck "Headers and footers"** to prevent blank pages
4. Click "Save" to generate the PDF

### Troubleshooting PDF Download Issues

If you experience issues with the PDF export (such as blank pages between slides):

1. **Use Chrome or Edge**: These browsers work best for PDF exports
2. **Disable headers and footers**: This is critical to prevent blank pages
3. **Set margins to none**: Minimizes whitespace in the output
4. **Try printing sections**: If the full presentation has issues, try printing specific sections
5. **Update your browser**: Older browser versions may have issues with the PDF export

## Repository Structure

```
lecture-series/
├── index.html                          # Main landing page
├── css/                                # Stylesheets
│   └── styles.css                      # Main CSS file
├── images/                             # Shared images (logos, etc.)
│   ├── gw_monogram_2c.png              # GWU logo (light mode)
│   └── gw_monogram_2c_rev.png          # GWU logo (dark mode)
├── lectures/                           # Lectures directory
│   └── 05-cache-coherence/             # Cache coherence lecture
│       ├── index.html                  # Lecture HTML
│       ├── slides.md                   # Lecture content in Markdown
│       └── images/                     # Lecture-specific images
│           └── *.svg                   # Diagrams and illustrations
├── .github/                            # GitHub configuration
│   └── workflows/                      # GitHub Actions workflows
│       └── deploy.yml                  # Automatic deployment config
├── start.sh                            # Script to start local server
└── README.md                           # This file
```

## Contributing

To add a new lecture:

1. Create a new folder with the lecture topic (e.g., `memory-consistency`)
2. Copy the structure from an existing lecture folder
3. Update the content in `slides.md`
4. Add any lecture-specific images to the lecture's `images` folder
5. Update the main `index.html` to include a link to the new lecture

## Deploying to GitHub Pages

This repository is configured for automatic deployment to GitHub Pages when changes are pushed to the main branch. The workflow configuration is in `.github/workflows/deploy.yml`.

To set up GitHub Pages:

1. Go to your repository settings
2. Navigate to "Pages" in the sidebar
3. Select the branch to deploy from (typically `gh-pages` or `main`)
4. Save changes

## License

[Specify appropriate license]

## Credits

- Reveal.js: https://revealjs.com/
- GWU ECE Department: https://www.ece.gwu.edu/ # GW-ECE6125-parallel-computer-architecture
