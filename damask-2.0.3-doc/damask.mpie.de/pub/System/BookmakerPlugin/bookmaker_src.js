/*
Author: Crawford Currie http://c-dot.co.uk

Copyright (C) 2011 Foswiki Contributors. Foswiki Contributors
are listed in the AUTHORS file in the root of this distribution.
NOTE: Please extend that file, not this notice.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version. For
more details read LICENSE in the root of this distribution.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

As per the GPL, removal of this notice is prohibited.
*/
Bookmaker = {
    jstree_options: {
	plugins: [ "json_data", "themes", "dnd", "crrm" ],
	dnd: {
	    drop_target: false,
	    drag_target: false
	},
	crrm_options: {
	    move: {}
	},
	themes: {
	    theme: "apple"
	}
    }
};

(function($) {

    $(document).ready(function() {
	$('#bookmaker_more').hide();
	$('#bookmaker_contract').click(function() {
	    $('#bookmaker_expand').fadeIn();
	    $('.bookmaker_active').fadeIn();
	    $('#bookmaker_more').slideUp();
	});
	$('.bookmaker_action').hide();
	$('.bookmaker_active').show();

	// If url is given, it is sent in place of the form data
	var sendForm = function(form, url) {
	    var url;
	    var data;
	    // Add the validation key
	    var k = form[0].validation_key ? StrikeOne.calculateNewKey(form[0].validation_key.value) : null;
	    if (!url) {
	        if (k)
		    form[0].validation_key = k;
		data = form.serialize();
		url = form.attr("action");
	    } else if (k) {
		data = { validation_key: k };
		// Split URL params into data
		var m = url.split('?');
		if (m.length > 1) {
		    url = m[0];
		    var ps = m[1].split(/[;&]/);
		    for (var pa in ps) {
			var p = ps[pa].split('=');
			data[p[0]] = p[1];
		    }
		}
	    }
	    // Send the request
	    $.ajax({
		type: form.attr("method"),
		url: url,
		data : data,
		dataType: "script",
		error: function(xhr, r, s) {
		    alert(r);
		}
	    });
	};

	$('#book_title').change(function() {
	    $("#new_book").fadeOut();
	    $("#bookmaker_change").fadeIn();
	    $(this).closest("form").submit();
	    return false;
	});

	$('#bookmaker_change').click(function(e) {
	    $('#bookmaker_change').fadeOut();
	    $('#new_book').fadeIn();
	});

	$('.bookmaker_button').click(function(e) {
	    sendForm($(this).closest("form"), $(this).attr("href"));
	    return false;
	});

	$("#book_tree").bind("move_node.jstree", function (e, data) {
	    // Drag-and-drop sorting
	    var url = $("#book_tree").jstree("get_settings").move_url;
	    data.rslt.o.each(function (i) {
		$.ajax({
		    async : false,
		    type: 'POST',
		    url: url,
		    data : { 
			what: $(this).attr("topic"),
			new_parent: data.rslt.np.attr("topic"),
			new_pos: data.rslt.cp + i
		    },
		    error: function(xhr, r, s) {
			console.debug("Move failed "+r);
			$.jstree.rollback(data.rlbk);
		    }
		});
	    });
	});
    });
})(jQuery);
