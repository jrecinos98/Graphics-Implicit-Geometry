 
// Making chess pieces using raymarching and constructive solid geometry
// Copy + paste all of this code into shadertoy and run it

// Defines
#define STEP_MAX 100
#define DIST_MAX 9.0
#define EPSILON 0.01

#define WHITE 0.
#define BLACK 1.
#define FLOOR 2.

// Get color based on which distance function was intersected with
vec3 get_color(float key){
    if(key == WHITE){
        return vec3(.55, .55, .55);
    } else if(key == BLACK){
        return vec3(.25, .25, .25);
    } else if(key == FLOOR){
        return vec3(.3,.3,.3);   
    } else{
        return vec3(.4, .2, .2);
    }
}

// Get material for phong reflectance 
vec3 get_mat(float key){
    if(key == WHITE || key == BLACK){
        return vec3(.8);
    }else{
        return vec3(.05);
    }
}

float sphere_sdf( vec3 p, float r)
{
  return length(p) - r;
}

// Signed distance functions defined here:
float sphere_sdf( vec3 p, vec3 c, float r)
{
  return length(p - c) - r;
}

// Distance function for the floor
float floor_sdf(vec3 p){
    vec4 plane = vec4(0, 1, 0, 0);
    return dot(p, plane.xyz) - plane.w;
}

float wall_sdf(vec3 p){
    vec4 plane = vec4(0, 0, -1, -9.0);
    return dot(p, plane.xyz) - plane.w;
}

float rounded_cylinder_sdf( vec3 p, float ra, float rb, float h )
{
  vec2 d = vec2( length(p.xz)- 2.0*ra+rb, abs(p.y) - h );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}
float rounded_cone_sdf( vec3 p, float r1, float r2, float h )
{
  vec2 q = vec2( length(p.xz), p.y );
    
  float b = (r1-r2)/h;
  float a = sqrt(1.0-b*b);
  float k = dot(q,vec2(-b,a));
    
  if( k < 0.0 ) return length(q) - r1;
  if( k > a*h ) return length(q-vec2(0.0,h)) - r2;
        
  return dot(q, vec2(a,b) ) - r1;
}



/**
 * Rotation matrix around the X axis.
 */
mat3 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(1, 0, 0),
        vec3(0, c, -s),
        vec3(0, s, c)
    );
}
/**
 * Rotation matrix around the Y axis.
 */
mat3 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, 0, s),
        vec3(0, 1, 0),
        vec3(-s, 0, c)
    );
}

/**
 * Rotation matrix around the Z axis.
 */
mat3 rotateZ(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, -s, 0),
        vec3(s, c, 0),
        vec3(0, 0, 1)
    );
}

float opSmoothUnion( float d1, float d2, float k ) {
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h); 
}

float opSmoothSubtraction( float d1, float d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h); }

float opSmoothIntersection( float d1, float d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) + k*h*(1.0-h); 
}
float opUnion( float d1, float d2 ) {  return min(d1,d2); }

vec2 opUnionVec2(vec2 d1, vec2 d2){
    if(d1.x < d2.x){
        return d1;
    }else{
        return d2;
    }
}

float opSubtraction( float d1, float d2 ) { return max(-d1,d2); }

float sdRoundedCylinder( vec3 p, float ra, float rb, float h, vec3 offset )
{
  p = p-offset;
  vec2 d = vec2( length(p.xz)-2.0*ra+rb, abs(p.y) - h );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}
float ndot(vec2 a, vec2 b ) { return a.x*b.x - a.y*b.y; }
float rhombus_sdf(vec3 p, float la, float lb, float h, float ra)
{
    p = abs(p);
    vec2 b = vec2(la,lb);
    float f = clamp( (ndot(b,b-2.0*p.xz))/dot(b,b), -1.0, 1.0 );
    vec2 q = vec2(length(p.xz-0.5*b*vec2(1.0-f,1.0+f))*sign(p.x*b.y+p.z*b.x-b.x*b.y)-ra, p.y-h);
    return min(max(q.x,q.y),0.0) + length(max(q,0.0));
}


float sdCappedCylinder( vec3 p, float h, float r, vec3 offset )
{
  p = p-offset;
  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdCappedCone(vec3 p, float h, float r1, float r2,vec3 offset)
{
  p = p - offset;
  vec2 q = vec2( length(p.xz), p.y );
  vec2 k1 = vec2(r2,h);
  vec2 k2 = vec2(r2-r1,2.0*h);
  vec2 ca = vec2(q.x-min(q.x,(q.y<0.0)?r1:r2), abs(q.y)-h);
  vec2 cb = q - k1 + k2*clamp( dot(k1-q,k2)/dot(k2,k2), 0.0, 1.0 );
  float s = (cb.x<0.0 && ca.y<0.0) ? -1.0 : 1.0;
  return s*sqrt( min(dot(ca,ca),dot(cb,cb)) );
}
float sdRoundBox( vec3 p, vec3 b, float r )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

float sdCone( vec3 p, vec2 c )
{
  // c is the sin/cos of the angle
  float q = length(p.xy);
  return dot(c,vec2(q,p.z));
}

float bishop_base(vec3 p){
    float base = rounded_cylinder_sdf(p, .45, .08, .18);
    float ring = rounded_cylinder_sdf(p - vec3(0,0.1,0), .25,.003,.08);
    return base;
   
}
float bishop_shaft(vec3 p){
    float ring = rounded_cylinder_sdf(p - vec3(0,2.5,0), .35,.005,.1);
    float ring2 = rounded_cylinder_sdf(p - vec3(0,2.7,0), .25,.003,.08);
    float ring3 = rounded_cylinder_sdf(p - vec3(0,2.9,0), .2,.001,.06);
    float shaft_tip = opUnion(opUnion(ring, ring2), ring3);   
    
    float cone = rounded_cone_sdf(p, .4,.1, 3.5);
    float ring4 = rounded_cylinder_sdf(p - vec3(0,0.4,0), .25,.003,.08);
    float shaft_base = opUnion(opSmoothUnion(cone, ring4, 0.3), ring4);
    return opSmoothUnion(shaft_base, shaft_tip,0.5);
}

float bishop_tip(vec3 p){
    float big_sphere = sphere_sdf(p, 0.50);
    float top_sphere = sphere_sdf(p - vec3(0,0.65,0),0.25);
    float spheres = opSmoothUnion(big_sphere, top_sphere, 0.1);
    vec3 rot = rotateZ(40.0) * (p - vec3(-0.55,0.35,0));
    float rhombus = rhombus_sdf(rot,0.5, 0.5, 0.08,0.4);
    //float t_prism = tri_prism_sdf(rot, vec2(0.55,0.5));
    return opSubtraction(rhombus,spheres);
    
}

vec2 bishop_sdf(vec3 point, vec3 offset, float color, float scale){
    vec3 p = point/scale;
    vec3 BISHOP_POS = offset/scale;
 
    float tip = bishop_tip(p-vec3(BISHOP_POS.x, BISHOP_POS.y + 5.0, BISHOP_POS.z));
    
    float base = bishop_base(p-vec3(BISHOP_POS.x, BISHOP_POS.y + 0.5, BISHOP_POS.z));
   
    float shaft = bishop_shaft(p - vec3(BISHOP_POS.x, BISHOP_POS.y + 1.0, BISHOP_POS.z));
    
    float peen = opSmoothUnion(opSmoothUnion(base, shaft,.65),tip,0.5);
    
    return vec2(peen*scale, color);
}

vec2 pawn(vec3 point, float base_h, float stem_h,  vec3 off, float color, float scale){
    vec3 p = point/scale;
    vec3 offset = off/scale;
    float init_base = sdCappedCylinder(p, 1.2,0.3, offset);
    offset = offset+vec3(0,0.3,0);
    float rounded_base = sdRoundedCylinder(p,0.7, 0.7, base_h, offset);
    offset = offset+vec3(0., 0.6, 0.);
    
    float flat_base = sdCappedCylinder(p, 1. ,0.1, offset);
    offset = offset+vec3(0., 0.1, 0.);
    float rounded_base2 = sdRoundedCylinder(p,0.5, 0.4, base_h, offset);
    offset = offset+vec3(0,0.25,0);
    float flat_base2 = sdCappedCylinder(p, 0.95 ,0.1, offset);
    offset = offset+vec3(0., 0.1, 0.);
    float base2 = opSmoothUnion(flat_base2, opSmoothUnion(flat_base,rounded_base2,0.1), 0.1);

    float stem  = sdCappedCone(p,stem_h,1.2, 0.25, offset);
    offset = offset+vec3(0., stem_h, 0.);
    float head = sphere_sdf(p, offset, 1.);
    float neck = sdCappedCylinder(p, 1.0,0.05, offset-vec3(0,1.0,0));

    float base = opSmoothUnion(init_base, opSmoothUnion(rounded_base,base2,0.01),0.01);
    float body = opUnion(neck, opUnion(stem,head));
    return vec2(opSmoothUnion( body, base, 0.1)*scale, color);     
}
float king_shaft(vec3 p){
    float ring = rounded_cylinder_sdf(p - vec3(0,3.1,0), .35,.005,.1);
    float ring2 = rounded_cylinder_sdf(p - vec3(0,3.3,0), .25,.003,.08);
    float ring3 = rounded_cylinder_sdf(p - vec3(0,3.5,0), .2,.003,.06);
    float shaft_tip = opUnion(opUnion(ring, ring2), ring3);   
    
    float cone = rounded_cone_sdf(p, .4,.07, 4.5);
    float ring4 = rounded_cylinder_sdf(p - vec3(0,0.4,0), .20,.003,.08);
    float shaft_base = opUnion(opSmoothUnion(cone, ring4, 0.3), ring4);
    return opSmoothUnion(shaft_base, shaft_tip,0.5);
}

float king_tip(vec3 p){
    float head = sdCappedCone(p,0.5, 0.1,0.55,vec3(0.,0.,0.));
    float h_rect1 = sdRoundBox(p-vec3(0.,0.8,0.), vec3(0.1,0.4,0.05),0.01);
    float h_rect2 = sdRoundBox(p-vec3(0., 0.9,0.), vec3(0.3,0.1,0.05),0.01);
    float cross =  opSmoothUnion(h_rect1, h_rect2, 0.07);
    return opUnion(head,cross);
   
}

vec2 king_sdf(vec3 point,vec3 offset, float color, float scale){
    vec3 p = point/scale;
    vec3 KING_POS = offset/scale;
    float base = bishop_base(p-vec3(KING_POS.x, KING_POS.y + 0.5, KING_POS.z));
   
    float shaft = king_shaft(p - vec3(KING_POS.x, KING_POS.y + 1.0, KING_POS.z));
    float head = king_tip(p-vec3(KING_POS.x, KING_POS.y + 5.5, KING_POS.z));
    return vec2(opSmoothUnion(opSmoothUnion(base, shaft,.65),head,0.5)*scale, color);
}


// Combine all the distance functions for the scene in this function
vec2 scene_sdf(vec3 p){
    vec2 curr = bishop_sdf(p, vec3(-2.,0,6), WHITE, 0.35);
    vec2 pawn = pawn(p, 0.009 , 3.50, vec3(2.,-0.0009, 6), WHITE, 0.2);
    vec2 king = king_sdf(p, vec3(0, 0, 6), BLACK, 0.4);
    vec2 plane = vec2(floor_sdf(p), 2); 
    vec2 wall = vec2(wall_sdf(p), 2);
    
    curr = opUnionVec2(curr, pawn);
    curr = opUnionVec2(curr, king);
    curr = opUnionVec2(curr, plane);
    curr = opUnionVec2(curr, wall);
    
    return curr;
}


// Perform ray marching by finding the min distance ray can travel without hitting anything and iterating
vec2 ray_march(vec3 cam_pos, vec3 cam_dir){
    float t_near = 0.0;
    for(int i = 0; i < STEP_MAX; i++){
        vec3 p = cam_pos + cam_dir * t_near; // t_near is how far we can go along ray without hitting object
        vec2 dist = scene_sdf(p);
        t_near += dist.x;
        // Check if we missed entirely or hit something
        // > DIST_MAX then we missed all objects, less than EPSILON, we hit an object 
        if(t_near > DIST_MAX){ 
            return vec2(-1., -1);
        }else if(dist.x < EPSILON){
            return vec2(t_near, dist.y); 
        }
    }
    
    return vec2(-1., -1);
}




// Get normal by approximating the gradient at some point in the scene
vec3 normal_at(vec3 p){
    float dist = scene_sdf(p).x;
    return normalize(dist - vec3( 
        scene_sdf(p - vec3(0.01, 0, 0)).x,
        scene_sdf(p - vec3(0, 0.01, 0)).x,
        scene_sdf(p - vec3(0, 0, 0.01)).x
    ));
}



// Add simple point lights to illuminate the scene
vec3 get_light(vec3 p, vec3 color, vec3 mat, vec3 cam_pos){
    vec3 l2_intensity = vec3(1.,.9,.8) * 2.;
    vec3 l2 = vec3(-1.5, 3, 4.);
    vec3 l2_dir = normalize(l2 - p);  // Direction vector from the point to light
    l2_dir.x = sin(iTime);
    float decay_l2 = (1. / length(p - l2));
    vec3 l_a = vec3(0.1);
    
    float p_s = 30.;
    
    vec3 norm = normal_at(p);  // Get the normal at the point
    
    // Coefficients for specular/diffuse
    vec3 kd = vec3(color);
    vec3 ks = vec3(mat);
    
    // Set up view direction
    vec3 v = cam_pos - p;
    v = normalize(v);
    
    //vec3 half_vec_l1 = (l1_dir + v) / (length(l1_dir + v));
    vec3 half_vec_l2 = (l2_dir + v) / (length(l2_dir + v));
    half_vec_l2 = normalize(half_vec_l2);

    
    
    float ndotl2 = dot(norm, l2_dir);  // Calculating dot for second light
    ndotl2 = clamp(ndotl2, 0.0, 1.0);
    
    float shadow = ray_march(p + norm * EPSILON * 2., l2_dir).x;  // MUST ADD Epsilon to ensure don't accidently hit the floor
    if(shadow < length(l2 - p) && shadow != -1.){  // Hit something between light and point so we're in a shadow
        l2_intensity *= .1;
    }
    color += kd * (ndotl2 * l2_intensity * decay_l2);
    //vec3 spec = pow(dot(half_vec_l1, norm),p_s) * ks * (l1_intensity * decay_l1);
    vec3 spec = pow(dot(half_vec_l2, norm), p_s) * ks * (l2_intensity * decay_l2);
    return color + l_a + spec;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = (fragCoord - (0.5) * iResolution.xy)/iResolution.y;

    vec3 cam_pos = vec3(0,1,0);
    vec3 cam_dir = vec3(uv.x, uv.y, 1);
    vec2 t = ray_march(cam_pos, cam_dir);
    vec3 col;
    if(t.x == -1.){
        col = get_color(t.y) * (1. - (uv.y));
    }else{
        vec3 point = cam_pos + cam_dir * t.x;  // Point in the scene (for shading purposes)
        vec3 new_col = get_light(point, get_color(t.y), get_mat(t.y), cam_pos);
        // Rendering to screen
        col = vec3(new_col);
    }
    fragColor = vec4(col,1.0);
}
