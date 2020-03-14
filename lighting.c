// Making chess pieces using raymarching and constructive solid geometry
// Copy + paste all of this code into shadertoy and run it

// Defines
#define STEP_MAX 200
#define DIST_MAX 100.0
#define EPSILON 0.001

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
	vec4 plane = vec4(0, 0, -1, -10);
    return dot(p, plane.xyz) - plane.w;
}

vec2 black_sphere_sdf(vec3 p, vec3 c, float r){
	return vec2(sphere_sdf(p, c, r), WHITE);   
}

// Combine all the distance functions for the scene in this function
vec2 scene_sdf(vec3 p){
    vec2 curr = black_sphere_sdf(p, vec3(0, 1, 8), 1.);
    vec2 plane = vec2(floor_sdf(p), 2); 
    vec2 wall = vec2(wall_sdf(p), 2);
    
    if(plane.x < curr.x){
    	curr = plane;   
    }
   	
    if(wall.x < curr.x){
        curr = wall;
    }
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
    vec3 l1_intensity = vec3(1.,.9,.8) * 4.;
    vec3 l1 = vec3(0, 7, 8);
    vec3 l1_dir = normalize(l1 - p);  // Direction vector from the point to light
    float decay_l1 = (1. / length(p - l1));
    
    vec3 l2_intensity = vec3(1.,.9,.8) * 4.;
    vec3 l2 = vec3(-2, .5, 4.);
    vec3 l2_dir = normalize(l2 - p);  // Direction vector from the point to light
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
    
    vec3 half_vec_l1 = (l1_dir + v) / (length(l1_dir + v));
    vec3 half_vec_l2 = (l2_dir + v) / (length(l2_dir + v));
    half_vec_l1 = normalize(half_vec_l1);
    half_vec_l2 = normalize(half_vec_l2);
    
    float ndotl1 = dot(norm, l1_dir);  // Calculate diffuse color intensity as dot product of light direction and surface normal
    ndotl1 = clamp(ndotl1, 0.0, 1.0);
    
    float shadow = ray_march(p + norm * EPSILON * 2., l1_dir).x;  // MUST ADD Epsilon to ensure don't accidently hit the floor
    if(shadow < length(l1 - p) && shadow != -1.){  // Hit something between light and point so we're in a shadow
    	l1_intensity = .2 * l1_intensity;
    }
    color += kd * (ndotl1 * (l1_intensity * decay_l1));
    
    
    float ndotl2 = dot(norm, l2_dir);  // Calculating dot for second light
    ndotl2 = clamp(ndotl2, 0.0, 1.0);
    
    shadow = ray_march(p + norm * EPSILON * 2., l2_dir).x;  // MUST ADD Epsilon to ensure don't accidently hit the floor
    if(shadow < length(l2 - p) && shadow != -1.){  // Hit something between light and point so we're in a shadow
    	l2_intensity *= .1;
    }
    color += kd * (ndotl2 * l2_intensity * decay_l2);
    vec3 spec = pow(dot(half_vec_l1, norm),p_s) * ks * (l1_intensity * decay_l1);
    spec += pow(dot(half_vec_l2, norm), p_s) * ks * (l2_intensity * decay_l2);
    return color + l_a + spec;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = (fragCoord - (0.5) * iResolution.xy)/iResolution.y;

    vec3 cam_pos = vec3(0,1,0);
    vec3 cam_dir = vec3(uv.x, uv.y, 1);
    vec3 col = vec3(0);
    vec2 t = ray_march(cam_pos, cam_dir);
    if(t.x == -1.){
        col = get_color(t.y) * (1. - (uv.y));
    }else{
        vec3 point = cam_pos + cam_dir * t.x;  // Point in the scene (for shading purposes)
        vec3 color = get_color(t.y);
        vec3 mat = get_mat(t.y);
        vec3 new_col = get_light(point, color, mat, cam_pos);
        // Rendering to screen
        col = vec3(new_col);
    }
    fragColor = vec4(col,1.0);
}