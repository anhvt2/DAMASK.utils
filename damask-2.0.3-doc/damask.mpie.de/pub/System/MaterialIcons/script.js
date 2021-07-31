"use strict";
jQuery(function($) {
   var timer;
   function filterOverview(val) {
      var regex = new RegExp(val, "i"),
         $overview = $(".overview");
      console.log("filterOverview(",val,")");
      if (val != '') {
         $overview.find(".entry").hide().each(function() {
            var $this = $(this);
            if (regex.test($this.text())) {
               $this.show();
            }
         });
      } else {
         $overview.find(".entry").show();
      }
   }
   $(".filterIcons:not(.filterIconsInited").livequery(function() {
      var $input = $(this);
      $input.addClass("filterIconsInited");
      $input.keyup(function() {
         if (typeof(timer) !== 'undefined') {
            //console.log("clearing timeout");
            window.clearTimeout(timer);
         }
         timer = window.setTimeout(function() {
            var val = $input.val();
            filterOverview(val);
            timer = undefined;
         }, 100);
         //console.log("new timer",timer);
      });
   });
});