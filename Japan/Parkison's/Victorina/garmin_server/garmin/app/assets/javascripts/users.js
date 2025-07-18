//= require moment
//= require moment-timezone-with-data
//= require tempusdominus-bootstrap-4
//= require moment/ja.js
//= require chartkick
//= require Chart.bundle
//= require filterrific/filterrific-jquery

$.fn.datetimepicker.Constructor.Default = $.extend({}, $.fn.datetimepicker.Constructor.Default, {
	icons: {
	    time: 'far fa-clock',
	    date: 'far fa-calendar',
	    up: 'fa fa-arrow-up',
	    down: 'fa fa-arrow-down',
	    previous: 'fa fa-chevron-left',
	    next: 'fa fa-chevron-right',
	    today: 'far fa-calendar-check-o',
	    clear: 'fa fa-trash',
	    close: 'fa fa-times'
	}
});

// $(document).on('turbolinks:load', function() {
// 	$(document).on('click', ".picker-switch.accordion-toggle a[data-action='clear']", function (e) {
// 		$(this).attr('data-turbolinks', 'false');		
// 	  // e.stopPropagation();
// 	});
// });

$(function () {
	$('#datetimepicker1').datetimepicker({
		useCurrent: true,
		timeZone: 'Asia/Tokyo',
		// buttons: {
		// 	showClear: true
		// },
		maxDate: moment.now(),
		ignoreReadonly: true,
		format: 'YYYY-MM-DD'
		// format: 'YYYY-MM-DD HH:mm ZZ'
	});
	$('#datetimepicker2').datetimepicker({
		useCurrent: true,
		timeZone: 'Asia/Tokyo',
		// buttons: {
		// 	showClear: true
		// },
		maxDate: moment.now(),
		ignoreReadonly: true,
		format: 'YYYY-MM-DD'
		// format: 'YYYY-MM-DD HH:mm ZZ'
  });
	$("#datetimepicker1").on("change.datetimepicker", function (e) {
    $('#datetimepicker2').datetimepicker('minDate', e.date);
  });
  $("#datetimepicker2").on("change.datetimepicker", function (e) {
    $('#datetimepicker1').datetimepicker('maxDate', e.date);
  });
  $('a[data-toggle="pill"]').on('shown.bs.tab', function (e) {
  	Chartkick.charts['sleep-chart-' + e.target.text.trim()].redraw();
	})
});