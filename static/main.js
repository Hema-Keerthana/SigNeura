function readPreview(input, imgEl) {
  if (!input.files || !input.files[0]) {
    imgEl.classList.add("d-none");
    return;
  }

  const reader = new FileReader();
  reader.onload = e => {
    imgEl.src = e.target.result;
    imgEl.classList.remove("d-none");
  };
  reader.readAsDataURL(input.files[0]);
}

document.addEventListener("DOMContentLoaded", () => {
  const originalInput = document.getElementById("originalInput");
  const testInput = document.getElementById("testInput");
  const originalPreview = document.getElementById("originalPreview");
  const testPreview = document.getElementById("testPreview");

  originalInput.addEventListener("change", () =>
    readPreview(originalInput, originalPreview)
  );
  testInput.addEventListener("change", () =>
    readPreview(testInput, testPreview)
  );
});
