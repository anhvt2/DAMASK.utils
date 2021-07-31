/*
 * jQuery.RedDot plugin
 *
 * Copyright (c) 2010-2016 Michael Daum http://michaeldaumconsulting.com
 *
 * Dual licensed under the MIT and GPL licenses:
 *   http://www.opensource.org/licenses/mit-license.php
 *   http://www.gnu.org/licenses/gpl.html
 *
 */
"use strict";
jQuery(function($) {

  /***************************************************************************
   * class definition
   */
  function RedDot(elem, opts) {
    var self = this;

    self.elem = $(elem);
    self.anchor = self.elem.children("a:first");
    self.opts = $.extend({}, opts, self.elem.data());
    self.init();
  }

  /***************************************************************************
   * init redDot instance
   */
  RedDot.prototype.init = function() {
    var self = this;

    if (typeof(self.opts.parent) === 'undefined') {
      self.parentElem = self.elem.parent();
    } else {
      self.parentElem = self.elem.parents(self.opts.parent).first();
    }

    self.parentElem.hoverIntent({
      over: function() {
        self.anchor.fadeIn(500, function() {
          self.anchor.css({opacity: 1.0});
        });
      },
      out: function() {
        self.anchor.stop();
        self.anchor.css({display:'none', opacity: 1.0});  
      }
    });
  };

  /***************************************************************************
   * make it a jQuery plugin
   */
  $.fn.redDot = function(opts) {
    return this.each(function() {
      if (!$.data(this, "redDot")) {
        $.data(this, "redDot", new RedDot(this, opts));
      }
    });
  };

  /***************************************************************************
   * enable declarative widget instanziation
   */
  $(".redDotAnimated:not(.redDotInited)").livequery(function() {
    $(this).addClass("redDotInited").redDot();
  });
});
