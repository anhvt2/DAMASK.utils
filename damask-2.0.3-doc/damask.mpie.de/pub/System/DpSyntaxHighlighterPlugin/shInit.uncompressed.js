(function($) {
  var brushes = [
        {
          aliases: ['applescript', 'as'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushAppleScript.js'
        },
        {
          aliases:['as3', 'actionScript3'], 
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushAS3.js'
        },
        {
          aliases: ['bash', 'shell', 'sh'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushBash.js'
        },
        {
          aliases: ['coldfusion', 'cf'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushColdFusion.js'
        },
        {
          aliases: ['cpp', 'cc', 'c++', 'c', 'h', 'hpp', 'h++'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushCpp.js'
        },
        {
          aliases: ['c#', 'c-sharp', 'csharp'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushCSharp.js'
        },
        {
          aliases: ['css'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushCss.js'
        },
        {
          aliases: ['delphi', 'pas', 'pascal'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushDelphi.js'
        },
        {
          aliases: ['diff', 'patch'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushDiff.js'
        },
        {
          aliases: ['erl', 'erlang'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushErlang.js'
        },
        {
          aliases: ['groovy'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushGroovy.js'
        },
        {
          aliases: ['haxe', 'hx'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushHaxe.js'
        },
        {
          aliases: ['javafx', 'jfx'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushJavaFX.js'
        },
        {
          aliases: ['java'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushJava.js'
        },
        {
          aliases: ['jscript', 'js', 'javascript', 'json'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushJScript.js'
        },
        {
          aliases: ['perl', 'pl'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushPerl.js'
        },
        {
          aliases: ['php'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushPhp.js'
        },
        {
          aliases: ['plain', 'text', 'txt', 'ascii'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushPlain.js'
        },
        {
          aliases: ['powershell', 'ps', 'posh'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushPowerShell.js'
        },
        {
          aliases: ['python', 'py'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushPython.js'
        },
        {
          aliases: ['ruby', 'rails', 'ror', 'rb'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushRuby.js'
        },
        {
          aliases: ['sass', 'sccs'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushSass.js'
        },
        {
          aliases: ['scala'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushScala.js'
        },
        {
          aliases: ['sql'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushSql.js'
        },
        {
          aliases: ['tap'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushTAP.js'
        },
        {
          aliases: ['ts', 'typescript'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushTypeScript.js'
        },
        {
          aliases: ['vb', 'vbnet', 'visualbasic' ],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushVb.js'
        },
        {
          aliases: ['xml', 'xhtml', 'xslt', 'html', 'plist'],
          url: 'DpSyntaxHighlighterPlugin/scripts/shBrushXml.js'
        }
      ], doneInit = false;

  function init() {
    if (doneInit) {
      return;
    }
    doneInit = true;
    
    var pubUrlPath = foswiki.getPreference("PUBURLPATH") + '/' + foswiki.getPreference("SYSTEMWEB"),
        i, l = brushes.length;

    for (i = 0; i < l; i++) {
      brushes[i].url = pubUrlPath + '/' + brushes[i].url;
    }
  }

  function getBrush(id) {
    var i, j, la, l = brushes.length;

    for (i = 0; i < l; i++) {
      la = brushes[i].aliases.length;
      for (j = 0; j < la; j++) {
        if (id == brushes[i].aliases[j]) {
          return brushes[i];
        }
      }
    }

    return;
  }

  function loadBrush(id, callback) {
    var brush = getBrush(id);

    if (!brush) {
      throw("no brush found for language '"+id+"'");
    }

    if (brush.def) {
      brush.def.then(callback);
    } else {
      brush.def = jQuery.Deferred();
      brush.def.then(callback);
      $.getScript(brush.url, function() { brush.def.resolve(); });
    }

  }

  function opts2params(opts) {
    var params = {};
    
    $.each(opts, function(key, val) {
      params[key.replace(/_/g,"-")] = val;
    });

    return params;
  }

  // dom loaded
  $(function() {
    init();

    $(".syntaxHighlight:not(.syntaxHighlightInited)").livequery(function() {
      var elem = this,
          $elem = $(elem),
          params = opts2params($elem.data());

       $elem.addClass("syntaxHighlightInited");      

       loadBrush(params.brush, function() {
          SyntaxHighlighter.highlight(params,elem);
       });
    });
  });
})(jQuery);
