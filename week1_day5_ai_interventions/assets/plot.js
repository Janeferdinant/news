// Synthetic data, written by ../regenerate_marks.py. Do not edit by hand.
document.addEventListener('DOMContentLoaded', function () {
  var data = [
    { y: [1, 4, -2, 1, 2, 5, 2, -1, 6, 2], type: 'box', name: 'Control' },
    { y: [2, 7, 5, 7, 3, 1, 4, 9, 2, 3], type: 'box', name: 'AI' }
  ];
  var layout = {
    title: 'Learning Gains by Group (synthetic data)',
    yaxis: { title: 'Score Gain' }
  };
  Plotly.newPlot('interactive-plot', data, layout);
});
