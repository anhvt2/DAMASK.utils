// C++ informative line for the emacs editor: -*- C++ -*-
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Board of Trustees of the University of Illinois.         *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the root of the source code       *
 * distribution tree, or in https://www.hdfgroup.org/licenses.               *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef __H5LinkCreatPropList_H
#define __H5LinkCreatPropList_H

namespace H5 {

/*! \class LinkCreatPropList
    \brief Class LinkCreatPropList inherits from StrCreatPropList and provides
    wrappers for the HDF5 link creation property list.
*/
// Inheritance: StrCreatPropList -> PropList -> IdComponent
class H5_DLLCPP LinkCreatPropList : public StrCreatPropList {
  public:
    ///\brief Default link creation property list.
    static const LinkCreatPropList &DEFAULT;

    // Creates a link creation property list.
    LinkCreatPropList();

    ///\brief Returns this class name.
    virtual H5std_string
    fromClass() const
    {
        return ("LinkCreatPropList");
    }

    // Copy constructor: same as the original LinkCreatPropList.
    LinkCreatPropList(const LinkCreatPropList &original);

    // Creates a copy of an existing link creation property list
    // using the property list id.
    LinkCreatPropList(const hid_t lcpl_id);

    // Noop destructor
    virtual ~LinkCreatPropList();

#ifndef DOXYGEN_SHOULD_SKIP_THIS

    // Deletes the global constant, should only be used by the library
    static void deleteConstants();

  private:
    static LinkCreatPropList *DEFAULT_;

    // Creates the global constant, should only be used by the library
    static LinkCreatPropList *getConstant();

#endif // DOXYGEN_SHOULD_SKIP_THIS
};     // end of LinkCreatPropList

} // namespace H5

#endif // __H5LinkCreatPropList_H
