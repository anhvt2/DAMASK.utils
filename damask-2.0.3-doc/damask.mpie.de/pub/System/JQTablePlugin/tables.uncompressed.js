/*
 * Copyright (C) Foswiki Contributors 2010-2014
 * Author: Crawford Currie http://c-dot.co.uk
 * Javascript implementation of TablePlugin
 * Uses the jQuery tablesorter plugin
 */
(function($) {
    var jqtp = {
        // Mappings for tableframe attributes
        tableframe: {
            'void':   'none',
            'above':  'solid none none none',
            'below':  'none none solid none',
            'lhs':    'none none none solid',
            'rhs':    'none solid none none',
            'hsides': 'solid none solid none',
            'vsides': 'none solid none solid',
            'box':    'solid',
            'border': 'solid'
        },

        defaultOpts: {
            debug: false,
            widgets: ['colorize']
        },

        // Move the data associated with an element marked with the class
        // 'jqtp_process' to the next TABLE tag. This is required because
        // %TABLE expands to a DIV which has to be associated with
        // a table before it is deleted from the DOM.
        move_table_data : function(elem) {
            var $this = $(elem),
                params = $this.data(),
                $table = $(jqtp.next_table(elem));

            $this.removeClass("jqtp_process");
            if (!$table.length) {
                return;
            }
            $table.data('jqtp_process', params);
            $this.remove();
        },
        
        // Find the next TABLE tag after the given element. This matches
        // a %TABLE with the following <table>
        next_table : function(el) {
            if (el === null) {
                return null;
            }
            do {
                // Zoom up
                while (el.nextSibling === null) {
                    if (el.parentNode === null) {
                        return null; // end of the road
                    } else {
                        el = el.parentNode;
                    }
                }
                // Next
                el = el.nextSibling;
                // Zoom down; stop and visit tables we pass on the way
                while (el.tagName != 'TABLE' && el.firstChild) {
                    el = el.firstChild;
                }
            } while (el && el.tagName != 'TABLE');
            return el;
        },

        // Add units to a number, for CSS
        unitify : function(n) {
            if (/^\d+$/.test(n)) {
                return n + "px";
            } else {
                return n;
            }
        },

        // Process a table found in the page body. Optional control
        // %TABLE control parameters will have been moved from the
        // preceding <div class=jqtp_process into data-jqtp_process
        // by move_table_data, but is not relied on; the function works
        // on all tables regardless of the existance of companion
        // jqtp_process parameters.
        process_table : function($table) {
            var p = $table.data('jqtp_process');
            if (p !== undefined) {
                if (p.id !== undefined) {
                    $table.id = p.id;
                }
                if (p.summary !== undefined) {
                    $table.summary = p.summary;
                }
                if (p.caption !== undefined) {
                    $table.append("<caption>" + p.caption + "</caption>");
                }
            }

            jqtp.simplify_head_and_foot($table, p);

            jqtp.process_rowspans($table);

            if (p !== undefined) {
                jqtp.process_colours($table, p);
                jqtp.add_borders($table, p);
                jqtp.adjust_layout($table, p);
                jqtp.align($table, p);

                if (p.sort == "on" && (p.disableallsort != "on")) {
                    $table.addClass("jqtp_sortable");
                }
            }
        },

        // Find cell-collapse marks (^) and assign a rowspan
        // to the first non-^ cell in the rows above. This does
        // embedded tables correctly too.
        process_rowspans : function($t) {
            var span = /^\s*\^\s*$/;

            $t.find("tr").each(
                function () {
                    $(this).find("td,th")
                        .filter(
                            function() {
                                return !$(this).hasClass("jqtpRowspanned")
                                    && span.test( $(this).text() );
                            })
                        .each(
                            function() {
                                var offset = $(this).prevAll().length,
                                    rb = $(this);
                                do {
                                    rb = rb.parent().prev()
                                        .children().eq(offset);
                                } while (rb.hasClass("jqtpRowspanned"));
                                if (rb.attr("rowspan") === undefined) {
                                    rb.attr("rowspan", 1);
                                }
                                rb.attr("rowspan",
                                        parseInt(rb.attr("rowspan"), 10) + 1);
                                $(this).addClass("jqtpRowspanned");
                            });
                });
            // Now chop out the spanned cells
            $t.find(".jqtpRowspanned").remove();
        },

        // try and pull out head and foot. The browser does this job
        // itself, at least for the head and the body - at least, Chrome
        // does and probably other webkit browsers too.
        // p provides advisory header and footer row counts (hrc and frc)
        simplify_head_and_foot : function($table, p) {
            if (!p) p = {};
            var hrc = p.headerrows,
                frc = p.footerrows;
            var $tbody, $thead, thcount, $children, headrows,
                $kid, tdcount, footrows, $tfoot;

            $tbody = $table.children('tbody');
            if ($tbody.length === 0) {
                // Browsers won't normally allow this to happen,
                // but just in case, if there's no tbody, create one
                $tbody = $('<tbody></tbody>');
                $thead = $table.children('THEAD');
                if (thead.length > 0) {
                    // stick it after the first thead
                    $tbody.insertAfter($thead.first());
                } else {
                    // ... or at the start of the table
                    $table.prepend($tbody);
                }
                // Move all top-level TR's into the tbody
                $tbody.append($table.children('TR').remove());
            }
            else {
                // Hope there's only one!
                $tbody = $tbody.first();
            }

            // Existance of a thead means hrc is ignored
            if ($table.children('thead').length === 0) {
                // No THEAD, but body may have rows containing TH's.
                // See how many.
                thcount = 0;
                $children = $tbody.children('TR');
                headrows = [];
                if (typeof(hrc) === 'undefined') {
                   hrc = $children.length;
                }
                while (thcount < hrc) {
                    $kid = $($children[thcount]);
                    if (!$kid.children().first().is('TH')) {
                        break;
                    }
                    headrows.push($kid);
                    thcount++;
                }

                if (hrc > thcount) {
                    hrc = thcount;
                }

                if (hrc > 0) {
                    $thead = $("<thead></thead>");
                    while (hrc--) {
                        $thead.append(headrows.shift().remove());
                    }
                    $table.prepend($thead);
                }
            }

            // There's a bug in Render.pm that makes it generate
            // an empty tfoot even if there are footer rows
            $table.children('tfoot')
                .filter(function() {
                    return ($(this).children().length === 0)
                })
                .remove();

            // Existance of a tfoot means frc is ignored
            if (frc !== undefined
                && $table.children('tfoot').length == 0) {

                // No TFOOT, but body may contain enough rows to make one
                tdcount = 0;
                $children = $tbody.children('TR');
                footrows = [];

                // Can't have more rows in the footer than exist
                if (frc > $children.length) {
                    frc = $children.length;
                }

                while (tdcount < frc) {
                    $kid = $($children[$children.length - 1 - tdcount]);
                    if (!$kid.children().first().is('TD,TH')) {
                        break;
                    }
                    footrows.push($kid);
                    tdcount++;
                }

                if (tdcount > 0) {
                    $tfoot = $("<tfoot></tfoot>");
                    while (frc--) {
                        $tfoot.append(footrows.pop().remove());
                    }
                    $table.append($tfoot);
                }
            }
        },

        // handle colour options
        process_colours : function($t, p) {
            var h,c,i;
            if (p.headerbg !== undefined || p.headercolor !== undefined) {
                h = $t.find('thead').add($t.find('tfoot'));
                if (h.length) {
                    if (p.headerbg !== undefined) {
                        h.css("background-color", p.headerbg);
                    }
                    if (p.headercolor !== undefined) {
                        h.css("color", p.headercolor);
                    }
                }
            }
            if (p.databg !== undefined || p.datacolor !== undefined) {
                h = $t.find('tbody > tr');
                if (h.length) {
                    if (p.databg !== undefined) {
                        c = p.databg.split(/\s*,\s*/);
                        for (i = 0; i < h.length; i++) {
                            $(h[i]).css("background-color",
                                        c[i % c.length]);
                        }
                    }
                    if (p.datacolor !== undefined) {
                        c = p.datacolor.split(/\s*,\s*/);
                        for (i = 0; i < h.length; i++) {
                            $(h[i]).css("color",
                                        c[i % c.length]);
                        }
                    }
                }
            }
        },

        // handle border options
        add_borders: function($t, p) {
            if (p.tableborder !== undefined) {
                $t[0].border = p.tableborder;
            }
            if (p.tableframe !== undefined
                && jqtp.tableframe[p.tableframe] !== undefined) {
                $t.css('border-style', jqtp.tableframe[p.tableframe]);
            }
            if (p.tablerules === undefined) {
                p.tablerules = "rows";
            }
            $t[0].rules = p.tablerules;
            if (p.cellborder !== undefined) {
                $t.find("td").add($t.find("th"))
                    .css("border-width", jqtp.unitify(p.cellborder));
            }
        },

        // handle layout options
        adjust_layout: function($t, p) {
            var h, cw;

            if (p.cellpadding !== undefined) {
                $t[0].cellPadding = p.cellpadding;
            }
            if (p.cellpadding !== undefined) {
                $t[0].cellSpacing = p.cellspacing;
            }
            if (p.tablewidth !== undefined) {
                $t[0].width = p.tablewidth;
            }
            if (p.columnwidths !== undefined) {
                cw = p.columnwidths.split(/\s*,\s*/);
                h = $t.find('tr').each(
                    function() {
                        var i = 0,
                            kid = this.firstChild, cs;

                        while (kid && i < cw.length) {
                            cs = kid.colSpan;
                            if (cs < 1) {
                              cs = 1;
                            }
                            // Skip columns with a colspan > 1
                            if (cs == 1) {
                                $(kid).css("width", jqtp.unitify(cw[i]));
                            }
                            i += cs;
                            do {
                                kid = kid.nextSibling;
                            } while (kid && kid.nodeType != 1);
                        }
                    });
            }
        },

        // handle alignment options
        align : function($t, p) {
            if (p.valign === undefined)
                p.valign = "top";
 
            if (p.headervalign === undefined) {
                p.headervalign = p.valign;
            }
            if (p.datavalign === undefined) {
                p.datavalign = p.valign;
            }

            if (p.headeralign !== undefined) {
                $t.find("thead > tr > th")
                    .add($t.find("thead > tr > td"))
                    .add($t.find("tfoot > tr > th"))
                    .add($t.find("tfoot > tr > td"))
                    .css("vertical-align", p.headervalign)
                    .css("text-align", p.headeralign);
            }

            if (p.dataalign !== undefined) {
                $t.find("tbody > tr > td")
                    .add($t.find("tbody > tr > th"))
                    .css("vertical-align", p.datavalign)
                    .css("text-align", p.dataalign);
            }
        },

        // handle sort options; cache them on the table for picking up when
        // we init tablesorter
        make_sortable: function(elem) {
            var sortOpts = $.extend({}, jqtp.defaultOpts),
            $elem = $(elem),
            p = $elem.data(),
            sortcol = [0, 0], 
            className, cols, col;

            if (p.initSort !== undefined) {
                sortcol[0] = p.initSort - 1;
                sortOpts.sortList = [sortcol];
            }
            if (p.initdirection !== undefined) {
                sortcol[1] = (p.initdirection == "down") ? 1 : 0;
                sortOpts.sortList = [sortcol];
            }

            if (p.databgsorted !== undefined) {

                className = 'jqtp_databgsorted_' +
                    p.databgsorted.replace(/\W/g, '_');

                /* Simplification; rather than pissing about colouring
                   alternate rows, paint all rows the same colour. */
                cols = p.databgsorted.split(/\s*,\s*/);
                col = cols[0];

                $("body").append('<style type="text/css">.' + className +
                                 '{background-color:' + col +
                                 '}</style>');
                sortOpts.cssAsc = className;
                sortOpts.cssDesc = className;
            }

            if (!$elem.find("thead").length) {
                jqtp.simplify_head_and_foot($elem);
            }

            $elem.tablesorter(sortOpts);
        }
    };

    // colorize columns
    $.tablesorter.addWidget({
        id: 'colorize',
        format: function(table) {
            $(".sorted", table).removeClass("sorted");
            $("th.headerSortDown, th.headerSortUp", table).each(function() {
                var index = this.cellIndex +1;
                $("td:nth-child("+index+")", table).addClass("sorted");
            });
        }
    });

    // add javascript date parser
    $.tablesorter.addParser({
        id: "date",
        is: function (s) {
            return !isNaN((new Date(s)).getTime());
        }, format: function (s) {
            return $.tablesorter.formatFloat(new Date(s).getTime());
        }, type: "numeric"
    });

    /// document ready
    $(function() {
        // Move %TABLE tag data to neighbouring TABLE
        $(".jqtp_process").livequery(function() {
            jqtp.move_table_data(this);
        });

        // Process all tables found in .foswikiContent (otherwise we
        // may screw up page headers / footers / menus)
        $('table.foswikiTable').livequery(function() {
            jqtp.process_table($(this));
        });

        // If sort is all, attach the sortable class to all tables
        var selector = ".jqtp_sortable",
           sort = foswiki.getPreference("JQTablePlugin.sort");

        if (sort) {
            if (sort == 'all') {
                selector += ", .foswikiTable:not(.foswikiTableInited)";
            } else if (sort == 'attachments') {
                // Just attachments
                selector += ", .foswikiAttachments:not(.foswikiAttachmentsInited) table";
            }
        }

        // Process tables marked as sortable
        $(selector).livequery(function() {
            $(this).addClass("jqtp_sortable");
            jqtp.make_sortable(this);
        });
    });

})(jQuery);
