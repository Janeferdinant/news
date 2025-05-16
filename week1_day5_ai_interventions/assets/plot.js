document.addEventListener('DOMContentLoaded', function () {
  var data = [
    { y: [2, 1, -1], type: 'box', name: 'Control' },
    { y: [10, 8, 8], type: 'box', name: 'AI' }
  ];
  var layout = {
    title: 'Learning Gains by Group',
    yaxis: { title: 'Score Gain' }
  };
  Plotly.newPlot('interactive-plot', data, layout);
});
