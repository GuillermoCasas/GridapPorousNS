const path = require('path');

const isRunFromRepomixFolder = path.basename(process.cwd()) === 'repomix';

module.exports = isRunFromRepomixFolder ? {
  // Config when run inside repomix/ directory
  output: {
    filePath: "repomix-output.xml",
    style: "xml",
    parsableStyle: true
  },
  include: [
    "../**/*"
  ],
  ignore: {
    useGitignore: true,
    useDefaultPatterns: true,
    customPatterns: [
      "repomix-output.*",
      "*.repomix-output.*",
      ".repomixignore",
      "**/*.pdf",
      "../.git/**/*",
      "../node_modules/**/*",
      "../scratch/**/*",
      "../**/results/**/*",
      "../**/*.msh",
      "../**/*.vtu",
      "../**/*.vtk",
      "../**/*.pvd",
      "../**/*.h5",
      "../**/*.json",
      "../theory/**/*",
      "../.agents/**/*"
    ]
  }
} : {
  // Config when run from the root directory
  output: {
    filePath: "repomix/repomix-output.xml",
    style: "xml",
    parsableStyle: true
  },
  include: [
    "**/*"
  ],
  ignore: {
    useGitignore: true,
    useDefaultPatterns: true,
    customPatterns: [
      "repomix-output.*",
      "*.repomix-output.*",
      ".repomixignore",
      "**/*.pdf",
      ".git/**/*",
      "node_modules/**/*",
      "scratch/**/*",
      "**/results/**/*",
      "**/*.msh",
      "**/*.vtu",
      "**/*.vtk",
      "**/*.pvd",
      "**/*.h5",
      "**/*.json",
      "theory/**/*",
      ".agents/**/*"
    ]
  }
};
